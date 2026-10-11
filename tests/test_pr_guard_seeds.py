"""
Tests for PR guard dbt seed CSV allowlist.
"""

from scripts.pr_guard import PRGuard


def test_dbt_seed_csv_allowed():
    """Test that small CSV files in dbt/seeds/ are allowed."""
    guard = PRGuard(github_token="fake", repo="owner/repo", pr_number=1)

    files = [
        {
            "filename": "dbt/seeds/team_map.csv",
            "status": "added",
            "additions": 50,
            "changes": 50,
            "patch": "+team,abbr\n+Chiefs,KC",
        }
    ]

    guard.check_changed_files(files)
    assert len(guard.errors) == 0


def test_dbt_seed_csv_too_large():
    """Test that large CSV files in dbt/seeds/ are rejected."""
    guard = PRGuard(github_token="fake", repo="owner/repo", pr_number=1)

    files = [
        {
            "filename": "dbt/seeds/huge_seed.csv",
            "status": "added",
            "additions": 5000,
            "changes": 5000,
        }
    ]

    guard.check_changed_files(files)
    assert len(guard.errors) == 1
    assert guard.errors[0].check == "file-size"
    assert "too large" in guard.errors[0].message
