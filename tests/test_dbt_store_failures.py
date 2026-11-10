"""Tests for the dbt store_failures quota check."""

from pathlib import Path

import pytest
import yaml

from scripts.check_dbt_store_failures import (
    MAX_STORED_FAILURE_RELATIONS,
    StoreFailureScan,
    main,
    scan_store_failures,
)

pytestmark = pytest.mark.cross_platform

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _write(dbt_dir: Path, relative: str, text: str) -> None:
    """Write a UTF-8 project file under ``dbt_dir``."""
    path: Path = dbt_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_cap_is_zero() -> None:
    """The project cap stays at zero stored-failure relations."""
    assert MAX_STORED_FAILURE_RELATIONS == 0


def test_real_project_stores_no_failure_relations() -> None:
    """This repo does not enable store_failures for any test."""
    scan: StoreFailureScan = scan_store_failures(REPO_ROOT / "dbt")
    assert scan.violations == ()
    assert scan.relation_count == 0


def test_real_project_keeps_the_audit_schema() -> None:
    """One-off --store-failures still lands in the test_results schema."""
    text: str = (REPO_ROOT / "dbt" / "dbt_project.yml").read_text(encoding="utf-8")
    payload: object = yaml.safe_load(text)
    assert isinstance(payload, dict)
    tests_config: object = payload["tests"]["endzone_mlops"]
    assert isinstance(tests_config, dict)
    assert tests_config["+schema"] == "test_results"
    assert "store_failures" not in tests_config
    assert "+store_failures" not in tests_config


def test_real_project_cli_passes() -> None:
    """The command-line checker accepts this repository."""
    status: int = main(["--root", str(REPO_ROOT)])
    assert status == 0


def test_project_level_true_is_broad(tmp_path: Path) -> None:
    """A project setting would store one relation per test."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "dbt_project.yml",
        "tests:\n  endzone_mlops:\n    +store_failures: true\n    +schema: test_results\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.relation_count == 0
    assert len(scan.violations) == 1
    assert "every test" in scan.violations[0]
    assert "view" in scan.violations[0]


def test_project_level_view_is_broad(tmp_path: Path) -> None:
    """Switching the project setting to views still hits the schema quota."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "dbt_project.yml",
        "data_tests:\n  +store_failures: view\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert any("view" in item and "every test" in item for item in scan.violations)


def test_directory_level_true_is_broad(tmp_path: Path) -> None:
    """A directory config covers every test in that directory."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "dbt_project.yml",
        "models:\n  endzone_mlops:\n    marts:\n      +store_failures: true\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.violations
    assert "every test" in scan.violations[0]


def test_false_and_schema_only_pass(tmp_path: Path) -> None:
    """An explicit false, and a schema with no flag, store nothing."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "dbt_project.yml",
        "tests:\n  endzone_mlops:\n    +store_failures: false\n    +schema: test_results\n",
    )
    _write(
        dbt_dir,
        "models/schema.yml",
        "models:\n  - name: fct_games\n    description: 'store_failures: true is prose, not a key'\n"
        "    columns:\n      - name: game_id\n        tests:\n          - not_null\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.violations == ()
    assert scan.relation_count == 0


def test_commented_flag_does_not_count(tmp_path: Path) -> None:
    """A commented flag is not configuration."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "dbt_project.yml",
        "# +store_failures: true\ntests:\n  endzone_mlops:\n    +schema: test_results\n",
    )
    _write(
        dbt_dir,
        "tests/assert_rows.sql",
        "-- store_failures=true\nselect 1 as game_id\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.violations == ()
    assert scan.relation_count == 0


def test_per_test_true_fails_when_allowlist_is_empty(tmp_path: Path) -> None:
    """One column test with store_failures on is still a stored relation."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "models/schema.yml",
        "models:\n  - name: fct_games\n    columns:\n      - name: game_id\n"
        "        tests:\n          - unique:\n              config:\n"
        "                store_failures: true\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.relation_count == 1
    assert any("models/schema.yml:fct_games.game_id.unique" in item for item in scan.violations)
    assert any("The cap is 0" in item for item in scan.violations)


def test_view_on_one_test_counts_as_a_relation(tmp_path: Path) -> None:
    """A view is a Unity Catalog relation, same quota as a table."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "models/schema.yml",
        "models:\n  - name: fct_games\n    tests:\n"
        "      - dbt_utils.expression_is_true:\n          config:\n"
        "              store_failures: view\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.relation_count == 1
    assert any("view" in item for item in scan.violations)


def test_repeated_tests_get_distinct_ids(tmp_path: Path) -> None:
    """Two expression tests on one model are not the same allowlist entry."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "models/schema.yml",
        "models:\n  - name: fct_games\n    tests:\n"
        "      - dbt_utils.expression_is_true:\n          config:\n"
        "              store_failures: true\n"
        "      - dbt_utils.expression_is_true:\n          config:\n"
        "              store_failures: true\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.relation_count == 2
    joined: str = "\n".join(scan.violations)
    assert "models/schema.yml:fct_games.dbt_utils.expression_is_true" in joined
    assert "models/schema.yml:fct_games.dbt_utils.expression_is_true#2" in joined


def test_allowlisted_test_under_the_cap_passes(tmp_path: Path) -> None:
    """A named test may store one relation when the cap allows it."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "models/schema.yml",
        "models:\n  - name: fct_games\n    columns:\n      - name: game_id\n"
        "        tests:\n          - unique:\n              config:\n"
        "                store_failures: true\n",
    )
    test_id: str = "models/schema.yml:fct_games.game_id.unique"
    scan: StoreFailureScan = scan_store_failures(
        dbt_dir,
        allowed_tests=frozenset({test_id}),
        max_relations=1,
    )
    assert scan.violations == ()
    assert scan.relation_count == 1


def test_allowlisted_tests_over_the_cap_fail(tmp_path: Path) -> None:
    """The cap still fails when every stored test is named."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "seeds/schema.yml",
        "seeds:\n  - name: known_pbp_gaps\n    columns:\n"
        "      - name: game_id\n        tests:\n          - not_null:\n"
        "              config:\n                store_failures: true\n"
        "      - name: team\n        tests:\n          - not_null:\n"
        "              config:\n                store_failures: table\n",
    )
    scan: StoreFailureScan = scan_store_failures(
        dbt_dir,
        allowed_tests=frozenset(
            {
                "seeds/schema.yml:known_pbp_gaps.game_id.not_null",
                "seeds/schema.yml:known_pbp_gaps.team.not_null",
            }
        ),
        max_relations=1,
    )
    assert scan.relation_count == 2
    assert any("The cap is 1" in item for item in scan.violations)
    assert not any("unless that test is named" in item for item in scan.violations)


def test_ephemeral_is_not_a_relation_and_still_needs_the_allowlist(tmp_path: Path) -> None:
    """Ephemeral stores no Unity Catalog object, and it is still the flag."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "models/schema.yml",
        "models:\n  - name: fct_games\n    tests:\n      - unique:\n"
        "          config:\n            store_failures: ephemeral\n",
    )
    blocked: StoreFailureScan = scan_store_failures(dbt_dir)
    assert blocked.relation_count == 0
    assert blocked.violations
    allowed: StoreFailureScan = scan_store_failures(
        dbt_dir,
        allowed_tests=frozenset({"models/schema.yml:fct_games.unique"}),
        max_relations=0,
    )
    assert allowed.violations == ()
    assert allowed.relation_count == 0


def test_singular_sql_true_is_one_relation(tmp_path: Path) -> None:
    """A singular test config(store_failures=true) is one stored relation."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "tests/assert_rows.sql",
        "{{ config(store_failures=true) }}\nselect 1 as game_id\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.relation_count == 1
    assert any("tests/assert_rows.sql" in item for item in scan.violations)


def test_singular_sql_false_passes(tmp_path: Path) -> None:
    """config(store_failures=false) stores nothing."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "tests/assert_rows.sql",
        "{{ config(\n    store_failures=false,\n    schema='test_results'\n) }}\nselect 1\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.violations == ()
    assert scan.relation_count == 0


def test_model_sql_config_is_broad(tmp_path: Path) -> None:
    """store_failures on a model file is not a single schema test."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(
        dbt_dir,
        "models/marts/fct_games.sql",
        "{{ config(materialized='table', store_failures=true) }}\nselect 1\n",
    )
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.relation_count == 0
    assert any("every test" in item for item in scan.violations)


def test_yaml_parse_error_fails(tmp_path: Path) -> None:
    """A file that cannot be parsed does not hide a later flag."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(dbt_dir, "dbt_project.yml", "tests: [\n")
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert any("could not be read as YAML" in item for item in scan.violations)


def test_cli_fails_on_a_broad_flag(tmp_path: Path) -> None:
    """The command-line checker returns 1 when the project flag is on."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(dbt_dir, "dbt_project.yml", "tests:\n  +store_failures: true\n")
    status: int = main(["--root", str(tmp_path)])
    assert status == 1


def test_profiles_and_package_lock_are_ignored(tmp_path: Path) -> None:
    """A local profile and the package lock are not project test config."""
    dbt_dir: Path = tmp_path / "dbt"
    _write(dbt_dir, "dbt_project.yml", "name: endzone_mlops\n")
    _write(dbt_dir, "profiles.yml", "tests:\n  +store_failures: true\n")
    _write(dbt_dir, "package-lock.yml", "tests:\n  +store_failures: true\n")
    _write(dbt_dir, "dbt_packages/dbt_utils/dbt_project.yml", "tests:\n  +store_failures: true\n")
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    assert scan.violations == ()
    assert scan.relation_count == 0
