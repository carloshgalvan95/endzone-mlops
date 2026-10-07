"""
Tests for PR guard script.

Tests validation logic against various PR scenarios.
"""

from unittest.mock import patch

import pytest

from scripts.pr_guard import PRGuard, ValidationError


class TestPRGuard:
    """Test PR guard validation checks."""

    @pytest.fixture
    def mock_pr_guard(self) -> PRGuard:
        """Create a mock PR guard instance."""
        return PRGuard(
            github_token="test_token",
            repo="carloshgalvan95/endzone-mlops",
            pr_number=1,
        )

    def test_correct_author_email(self, mock_pr_guard: PRGuard) -> None:
        """Valid author email passes check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "author": {"email": "carloshgalvan95@gmail.com"},
                    "committer": {"email": "carloshgalvan95@gmail.com"},
                    "message": "feat: add feature",
                },
                "author": {"login": "carloshgalvan95"},
                "committer": {"login": "carloshgalvan95"},
                "parents": [{"sha": "parent1"}],
            }
        ]

        mock_pr_guard.check_commit_authorship(commits)
        assert len(mock_pr_guard.errors) == 0

    def test_wrong_author_email(self, mock_pr_guard: PRGuard) -> None:
        """Wrong author email fails check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "author": {"email": "wrong@example.com"},
                    "committer": {"email": "carloshgalvan95@gmail.com"},
                    "message": "feat: add feature",
                },
                "author": {"login": "carloshgalvan95"},
                "committer": {"login": "carloshgalvan95"},
                "parents": [{"sha": "parent1"}],
            }
        ]

        mock_pr_guard.check_commit_authorship(commits)
        assert len(mock_pr_guard.errors) == 1
        assert mock_pr_guard.errors[0].check == "commit-authorship"
        assert "wrong author email" in mock_pr_guard.errors[0].message.lower()

    def test_wrong_author_login(self, mock_pr_guard: PRGuard) -> None:
        """Wrong author login fails check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "author": {"email": "carloshgalvan95@gmail.com"},
                    "committer": {"email": "carloshgalvan95@gmail.com"},
                    "message": "feat: add feature",
                },
                "author": {"login": "wronguser"},
                "committer": {"login": "carloshgalvan95"},
                "parents": [{"sha": "parent1"}],
            }
        ]

        mock_pr_guard.check_commit_authorship(commits)
        assert len(mock_pr_guard.errors) == 1
        assert "wrong author login" in mock_pr_guard.errors[0].message.lower()

    def test_merge_commit_web_flow_allowed(self, mock_pr_guard: PRGuard) -> None:
        """GitHub web-flow committer allowed for merge commits."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "author": {"email": "carloshgalvan95@gmail.com"},
                    "committer": {"email": "noreply@github.com"},
                    "message": "Merge pull request #1",
                },
                "author": {"login": "carloshgalvan95"},
                "committer": {"login": "web-flow"},
                "parents": [{"sha": "parent1"}, {"sha": "parent2"}],
            }
        ]

        mock_pr_guard.check_commit_authorship(commits)
        assert len(mock_pr_guard.errors) == 0

    def test_co_authored_by_non_owner_fails(self, mock_pr_guard: PRGuard) -> None:
        """Co-authored-by trailer with non-owner email fails check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "message": "feat: add feature\n\nCo-authored-by: Someone <someone@example.com>"
                },
            }
        ]

        mock_pr_guard.check_commit_messages(commits)
        assert len(mock_pr_guard.errors) == 1
        assert mock_pr_guard.errors[0].check == "commit-message"
        assert "co-authored-by" in mock_pr_guard.errors[0].message.lower()
        assert "non-owner email" in mock_pr_guard.errors[0].message.lower()

    def test_co_authored_by_bot_fails(self, mock_pr_guard: PRGuard) -> None:
        """Co-authored-by trailer with bot email fails check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "message": "feat: add feature\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>"
                },
            }
        ]

        mock_pr_guard.check_commit_messages(commits)
        assert len(mock_pr_guard.errors) >= 1
        assert any(
            "co-authored-by" in e.message.lower() and "non-owner email" in e.message.lower()
            for e in mock_pr_guard.errors
        )

    def test_co_authored_by_self_passes(self, mock_pr_guard: PRGuard) -> None:
        """Self Co-authored-by trailer (owner's email) passes check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "message": "feat: add feature\n\nCo-authored-by: Carlos Galván <carloshgalvan95@gmail.com>"
                },
            }
        ]

        mock_pr_guard.check_commit_messages(commits)
        # Should have no errors for self Co-authored-by
        assert len(mock_pr_guard.errors) == 0

    def test_co_authored_by_mixed_fails(self, mock_pr_guard: PRGuard) -> None:
        """Mixed Co-authored-by trailers (self + other) fails check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {
                    "message": "feat: add feature\n\n"
                    "Co-authored-by: Carlos Galván <carloshgalvan95@gmail.com>\n"
                    "Co-authored-by: Someone Else <other@example.com>"
                },
            }
        ]

        mock_pr_guard.check_commit_messages(commits)
        assert len(mock_pr_guard.errors) == 1
        assert "non-owner email" in mock_pr_guard.errors[0].message.lower()

    def test_ai_mention_fails(self, mock_pr_guard: PRGuard) -> None:
        """AI agent mention in commit message fails check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {"message": "feat: add feature with cursor agent help"},
            }
        ]

        mock_pr_guard.check_commit_messages(commits)
        assert len(mock_pr_guard.errors) == 1
        assert "mentions ai/agent" in mock_pr_guard.errors[0].message.lower()

    def test_ai_mention_in_code_context_passes(self, mock_pr_guard: PRGuard) -> None:
        """AI mention in code context (db.cursor) passes check."""
        commits = [
            {
                "sha": "abc1234567890",
                "commit": {"message": "fix: close db.cursor() properly"},
            }
        ]

        mock_pr_guard.check_commit_messages(commits)
        assert len(mock_pr_guard.errors) == 0

    def test_conventional_commit_title_passes(self, mock_pr_guard: PRGuard) -> None:
        """Valid conventional commit titles pass."""
        valid_titles = [
            "feat: add new feature",
            "fix(auth): resolve login issue",
            "docs: update readme",
            "chore: upgrade dependencies",
            "ci: add pr guard workflow",
        ]

        for title in valid_titles:
            mock_pr_guard.errors = []
            pr = {"title": title}
            mock_pr_guard.check_pr_title(pr)
            assert len(mock_pr_guard.errors) == 0, f"Title '{title}' should pass"

    def test_non_conventional_title_fails(self, mock_pr_guard: PRGuard) -> None:
        """Non-conventional commit title fails check."""
        invalid_titles = [
            "Add new feature",
            "FIX: something",
            "feature: add something",
        ]

        for title in invalid_titles:
            mock_pr_guard.errors = []
            pr = {"title": title}
            mock_pr_guard.check_pr_title(pr)
            assert len(mock_pr_guard.errors) == 1, f"Title '{title}' should fail"
            assert "not a conventional commit" in mock_pr_guard.errors[0].message.lower()

    def test_pr_body_tool_footer_fails(self, mock_pr_guard: PRGuard) -> None:
        """Tool-generated footer in PR body fails check."""
        pr = {"body": "This is a PR\n\n<!-- CURSOR_AGENT_PR_BODY_BEGIN -->\nGenerated content"}

        mock_pr_guard.check_pr_body(pr)
        assert len(mock_pr_guard.errors) == 1
        assert "tool-generated footer" in mock_pr_guard.errors[0].message.lower()

    def test_pr_body_em_dash_fails(self, mock_pr_guard: PRGuard) -> None:
        """Em dash in PR body fails check."""
        pr = {"body": "This is a PR \u2014 with em dash"}

        mock_pr_guard.check_pr_body(pr)
        assert len(mock_pr_guard.errors) == 1
        assert "em dash" in mock_pr_guard.errors[0].message.lower()

    def test_pr_template_missing_section_fails(self, mock_pr_guard: PRGuard) -> None:
        """Missing template section fails check."""
        pr = {
            "body": "## Summary\nSome text\n\n## Changes\nMore text\n\n## Verification\n- [x] Done"
        }

        mock_pr_guard.check_pr_template(pr)
        assert len(mock_pr_guard.errors) == 1
        assert "missing required section: risks" in mock_pr_guard.errors[0].message.lower()

    def test_pr_template_unchecked_verification_fails(self, mock_pr_guard: PRGuard) -> None:
        """Entirely unchecked verification fails check."""
        pr = {
            "body": "## Summary\nText\n\n## Changes\nText\n\n## Verification\n- [ ] Item 1\n- [ ] Item 2\n\n## Risks\nNone"
        }

        mock_pr_guard.check_pr_template(pr)
        assert len(mock_pr_guard.errors) == 1
        assert (
            "verification checklist is entirely unchecked"
            in mock_pr_guard.errors[0].message.lower()
        )

    def test_pr_template_valid_passes(self, mock_pr_guard: PRGuard) -> None:
        """Valid PR template passes check."""
        pr = {
            "body": "## Summary\nText\n\n## Changes\nText\n\n## Verification\n- [x] Done\n- [ ] Optional\n\n## Risks\nNone"
        }

        mock_pr_guard.check_pr_template(pr)
        assert len(mock_pr_guard.errors) == 0

    def test_changed_file_em_dash_fails(self, mock_pr_guard: PRGuard) -> None:
        """Added em dash in changed file fails check."""
        files = [
            {
                "filename": "README.md",
                "status": "modified",
                "patch": "@@ -1,1 +1,1 @@\n-Old line\n+New line \u2014 with em dash",
            }
        ]

        mock_pr_guard.check_changed_files(files)
        assert len(mock_pr_guard.errors) == 1
        assert "em dash" in mock_pr_guard.errors[0].message.lower()

    def test_blocked_file_pattern_fails(self, mock_pr_guard: PRGuard) -> None:
        """Blocked file pattern fails check."""
        files = [
            {
                "filename": "data.parquet",
                "status": "added",
                "changes": 100,
                "additions": 100,
            },
            {
                "filename": ".env",
                "status": "added",
                "changes": 10,
                "additions": 10,
            },
        ]

        mock_pr_guard.check_changed_files(files)
        assert len(mock_pr_guard.errors) == 2
        assert any("blocked pattern" in e.message.lower() for e in mock_pr_guard.errors)

    def test_allowlisted_file_passes(self, mock_pr_guard: PRGuard) -> None:
        """Allowlisted file passes check."""
        files = [
            {
                "filename": "tests/fixtures/sample.csv",
                "status": "added",
                "changes": 10,
                "additions": 10,
            }
        ]

        mock_pr_guard.check_changed_files(files)
        assert len(mock_pr_guard.errors) == 0

    def test_base_branch_freshness_up_to_date(self, mock_pr_guard: PRGuard) -> None:
        """Up-to-date PR passes freshness check."""
        pr = {
            "base": {"sha": "base123", "ref": "main"},
            "head": {"sha": "head456"},
        }

        with patch.object(mock_pr_guard, "_get") as mock_get:
            mock_get.return_value = {"merge_base_commit": {"sha": "base123"}}
            mock_pr_guard.check_base_branch_freshness(pr)

        assert len(mock_pr_guard.errors) == 0

    def test_base_branch_freshness_stale(self, mock_pr_guard: PRGuard) -> None:
        """Stale PR fails freshness check."""
        pr = {
            "base": {"sha": "base123", "ref": "main"},
            "head": {"sha": "head456"},
        }

        with patch.object(mock_pr_guard, "_get") as mock_get:
            mock_get.return_value = {"merge_base_commit": {"sha": "oldbase"}}
            mock_pr_guard.check_base_branch_freshness(pr)

        assert len(mock_pr_guard.errors) == 1
        assert "behind base branch" in mock_pr_guard.errors[0].message.lower()


class TestValidationError:
    """Test ValidationError dataclass."""

    def test_validation_error_creation(self) -> None:
        """ValidationError can be created with check and message."""
        error = ValidationError(check="test-check", message="Test error message")
        assert error.check == "test-check"
        assert error.message == "Test error message"
