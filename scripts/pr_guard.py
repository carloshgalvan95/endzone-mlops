#!/usr/bin/env python3
"""
PR Guard: Automated guardrails for pull requests.

Catches common issues:
- Stale base branch (PR not up to date with base)
- Incorrect author/committer identity
- Co-authored-by trailers or AI agent mentions
- Non-conventional commit PR titles
- Tool-generated footers in PR body
- Em dash character (U+2014) in PR body or changed files
- Data/secret files in changes
- Missing PR template sections or unchecked verification
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ValidationError:
    """A single validation failure."""

    check: str
    message: str


class PRGuard:
    """Validates pull requests against repository quality standards."""

    EXPECTED_AUTHOR_EMAIL = "carloshgalvan95@gmail.com"
    EXPECTED_AUTHOR_LOGIN = "carloshgalvan95"
    GITHUB_WEB_FLOW_EMAIL = "noreply@github.com"

    # Conventional commit types
    CONVENTIONAL_COMMIT_TYPES = {
        "feat",
        "fix",
        "docs",
        "chore",
        "refactor",
        "test",
        "ci",
        "build",
        "perf",
    }

    # Tool-generated footer patterns
    TOOL_FOOTER_PATTERNS = [
        r"CURSOR_AGENT_PR_BODY",
        r"cursor\.com/agents",
        r"Open in Cursor",
        r"Open in Web",
    ]

    # AI agent mention patterns (word boundary, case-insensitive)
    AI_MENTION_PATTERNS = [
        r"\bcursor\b",
        r"\bcursoragent\b",
        r"\bclaude\b",
        r"\bcopilot\b",
        r"\bai agent\b",
        r"\bgrok\b",
    ]

    # File patterns to block (case-insensitive)
    BLOCKED_FILE_PATTERNS = [
        r"\.parquet$",
        r"\.csv$",
        r"\.csv\.gz$",
        r"^\.env$",
        r"\.pem$",
    ]

    # Allowlisted paths for data files
    DATA_FILE_ALLOWLIST = [
        r"^tests/fixtures/",
    ]

    # Max file size (bytes)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

    def __init__(self, github_token: str, repo: str, pr_number: int) -> None:
        """
        Initialize PR guard.

        Args:
            github_token: GitHub API token
            repo: Repository in owner/name format
            pr_number: Pull request number
        """
        self.github_token = github_token
        self.repo = repo
        self.pr_number = pr_number
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.errors: list[ValidationError] = []

    def _get(self, endpoint: str) -> Any:
        """Make a GET request to GitHub API."""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _get_pr(self) -> dict[str, Any]:
        """Fetch pull request details."""
        return self._get(f"/repos/{self.repo}/pulls/{self.pr_number}")

    def _get_commits(self) -> list[dict[str, Any]]:
        """Fetch all commits in the pull request."""
        return self._get(f"/repos/{self.repo}/pulls/{self.pr_number}/commits")

    def _get_files(self) -> list[dict[str, Any]]:
        """Fetch all files changed in the pull request."""
        return self._get(f"/repos/{self.repo}/pulls/{self.pr_number}/files")

    def _add_error(self, check: str, message: str) -> None:
        """Record a validation error."""
        self.errors.append(ValidationError(check=check, message=message))

    def check_base_branch_freshness(self, pr: dict[str, Any]) -> None:
        """
        Verify PR head contains the current base branch head.

        Ensures the PR is up to date with the base branch.
        """
        base_sha = pr["base"]["sha"]
        head_sha = pr["head"]["sha"]

        # Get merge base
        compare = self._get(f"/repos/{self.repo}/compare/{base_sha}...{head_sha}")

        # Check if base is reachable from head
        if compare["merge_base_commit"]["sha"] != base_sha:
            self._add_error(
                "base-branch-freshness",
                f"PR is behind base branch. Base SHA {base_sha[:7]} is not in "
                f"PR history. Please rebase: `git fetch && git rebase origin/{pr['base']['ref']}`",
            )

    def check_commit_authorship(self, commits: list[dict[str, Any]]) -> None:
        """
        Verify all commits have correct author and committer identity.

        Author and committer email must be carloshgalvan95@gmail.com.
        GitHub login must be carloshgalvan95 (or web-flow for merge commits).
        """
        for commit in commits:
            commit_sha = commit["sha"][:7]
            commit_msg_first_line = commit["commit"]["message"].split("\n")[0]

            # Check author email
            author_email = commit["commit"]["author"]["email"]
            if author_email != self.EXPECTED_AUTHOR_EMAIL:
                self._add_error(
                    "commit-authorship",
                    f"Commit {commit_sha} ({commit_msg_first_line}) has wrong "
                    f"author email: {author_email}. Expected: {self.EXPECTED_AUTHOR_EMAIL}",
                )

            # Check committer email (allow GitHub web-flow for merge commits)
            committer_email = commit["commit"]["committer"]["email"]
            is_merge_commit = len(commit.get("parents", [])) > 1
            if committer_email != self.EXPECTED_AUTHOR_EMAIL and not (
                is_merge_commit and committer_email == self.GITHUB_WEB_FLOW_EMAIL
            ):
                self._add_error(
                    "commit-authorship",
                    f"Commit {commit_sha} ({commit_msg_first_line}) has wrong "
                    f"committer email: {committer_email}. Expected: {self.EXPECTED_AUTHOR_EMAIL}",
                )

            # Check author GitHub login
            if commit["author"]:
                author_login = commit["author"]["login"]
                if author_login != self.EXPECTED_AUTHOR_LOGIN:
                    self._add_error(
                        "commit-authorship",
                        f"Commit {commit_sha} ({commit_msg_first_line}) has wrong "
                        f"author login: {author_login}. Expected: {self.EXPECTED_AUTHOR_LOGIN}",
                    )

            # Check committer GitHub login (allow web-flow for merge commits)
            if commit["committer"]:
                committer_login = commit["committer"]["login"]
                if committer_login != self.EXPECTED_AUTHOR_LOGIN and not (
                    is_merge_commit and committer_login == "web-flow"
                ):
                    self._add_error(
                        "commit-authorship",
                        f"Commit {commit_sha} ({commit_msg_first_line}) has wrong "
                        f"committer login: {committer_login}. Expected: {self.EXPECTED_AUTHOR_LOGIN}",
                    )

    def check_commit_messages(self, commits: list[dict[str, Any]]) -> None:
        """
        Verify commit messages don't contain Co-authored-by or AI mentions.

        Checks for:
        - Co-authored-by: trailer with email other than carloshgalvan95@gmail.com
        - Mentions of cursor, cursoragent, claude, copilot, AI agent, grok
        """
        for commit in commits:
            commit_sha = commit["sha"][:7]
            commit_msg = commit["commit"]["message"]

            # Check for Co-authored-by with non-owner email
            # Self Co-authored-by (Carlos's email) is allowed
            coauthor_pattern = r"^Co-authored-by:\s*(.+?)\s*<(.+?)>$"
            for match in re.finditer(coauthor_pattern, commit_msg, re.MULTILINE | re.IGNORECASE):
                email = match.group(2).strip()
                if email.lower() != self.EXPECTED_AUTHOR_EMAIL.lower():
                    self._add_error(
                        "commit-message",
                        f"Commit {commit_sha} contains Co-authored-by trailer with "
                        f"non-owner email: {email}. Only self Co-authored-by "
                        f"({self.EXPECTED_AUTHOR_EMAIL}) is allowed.",
                    )

            # Check for AI agent mentions (avoid false positives like DB cursor)
            for pattern in self.AI_MENTION_PATTERNS:
                # Only match in commit message, not in code/diffs
                matches = re.finditer(pattern, commit_msg, re.IGNORECASE)
                for match in matches:
                    # Skip if it's part of a larger word or code context
                    context = commit_msg[max(0, match.start() - 10) : match.end() + 10]
                    if not any(
                        skip in context.lower() for skip in ["cursor()", ".cursor", "db.cursor"]
                    ):
                        self._add_error(
                            "commit-message",
                            f"Commit {commit_sha} mentions AI/agent ({match.group()}). "
                            "Remove AI agent references from commit messages.",
                        )
                        break

    def check_pr_title(self, pr: dict[str, Any]) -> None:
        """
        Verify PR title follows Conventional Commit format.

        Expected format: type(optional-scope): description
        """
        title = pr["title"]

        # Pattern: type(optional scope): description
        pattern = r"^(" + "|".join(self.CONVENTIONAL_COMMIT_TYPES) + r")(\([a-z0-9\-]+\))?:\s+.+"
        if not re.match(pattern, title):
            self._add_error(
                "pr-title",
                f"PR title is not a Conventional Commit: '{title}'. "
                f"Expected format: type(scope): description. "
                f"Valid types: {', '.join(sorted(self.CONVENTIONAL_COMMIT_TYPES))}",
            )

    def check_pr_body(self, pr: dict[str, Any]) -> None:
        """
        Verify PR body doesn't contain tool footers or em dash.

        Also checks for AI mentions and em dash character (U+2014).
        """
        body = pr.get("body") or ""

        # Check for tool-generated footers
        for pattern in self.TOOL_FOOTER_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE):
                self._add_error(
                    "pr-body",
                    f"PR body contains tool-generated footer: {pattern}. "
                    "Remove all tool-generated markers.",
                )

        # Check for em dash (U+2014)
        if "\u2014" in body:
            self._add_error(
                "pr-body",
                "PR body contains em dash character (U+2014). "
                "Use comma, period, colon, or regular hyphen instead.",
            )

        # Check for AI mentions (same patterns as commit messages)
        for pattern in self.AI_MENTION_PATTERNS:
            matches = re.finditer(pattern, body, re.IGNORECASE)
            for match in matches:
                context = body[max(0, match.start() - 10) : match.end() + 10]
                if not any(
                    skip in context.lower() for skip in ["cursor()", ".cursor", "db.cursor"]
                ):
                    self._add_error(
                        "pr-body",
                        f"PR body mentions AI/agent ({match.group()}). Remove AI agent references.",
                    )
                    break

    def check_pr_template(self, pr: dict[str, Any]) -> None:
        """
        Verify PR body contains required template sections.

        Required sections: Summary, Changes, Verification, Risks
        Verification checklist must have at least one item checked.
        """
        body = pr.get("body") or ""

        # Check for required sections
        required_sections = ["Summary", "Changes", "Verification", "Risks"]
        for section in required_sections:
            # Case-insensitive heading check (Markdown ## heading or **bold**)
            if not re.search(
                rf"^##\s+{section}|^\*\*{section}\*\*",
                body,
                re.MULTILINE | re.IGNORECASE,
            ):
                self._add_error(
                    "pr-template",
                    f"PR body missing required section: {section}. Use the PR template.",
                )

        # Check verification checklist has at least one checked item
        # Pattern: - [x] or - [X]
        if not re.search(r"^- \[[xX]\]", body, re.MULTILINE):
            self._add_error(
                "pr-template",
                "PR body verification checklist is entirely unchecked. "
                "Complete verification steps and check at least one item.",
            )

    def check_changed_files(self, files: list[dict[str, Any]]) -> None:
        """
        Verify changed files don't add em dash or data/secret files.

        Checks for:
        - Em dash (U+2014) in text files
        - Data files (*.parquet, *.csv, *.csv.gz)
        - Secret files (.env, *.pem)
        - Large files (> 5 MB)
        """
        for file in files:
            filename = file["filename"]

            # Skip deleted files
            if file["status"] == "removed":
                continue

            # Check file size
            if file.get("changes", 0) > 0 or file.get("additions", 0) > 0:
                # Estimate file size from changes (rough heuristic)
                if file.get("changes", 0) > self.MAX_FILE_SIZE / 100:
                    self._add_error(
                        "file-size",
                        f"File {filename} appears very large (>{self.MAX_FILE_SIZE // (1024 * 1024)} MB). "
                        "Avoid committing large data files.",
                    )

            # Check for blocked file patterns (unless allowlisted)
            is_allowlisted = any(
                re.search(pattern, filename) for pattern in self.DATA_FILE_ALLOWLIST
            )

            if not is_allowlisted:
                for pattern in self.BLOCKED_FILE_PATTERNS:
                    if re.search(pattern, filename, re.IGNORECASE):
                        self._add_error(
                            "blocked-file",
                            f"File {filename} matches blocked pattern {pattern}. "
                            "Data and secret files should not be committed. "
                            f"Allowlist: {', '.join(self.DATA_FILE_ALLOWLIST)}",
                        )
                        break

            # Check patch for em dash in text files
            if file.get("patch"):
                patch = file["patch"]
                # Only check added lines (start with +)
                for line in patch.split("\n"):
                    if line.startswith("+") and "\u2014" in line:
                        self._add_error(
                            "em-dash",
                            f"File {filename} adds em dash character (U+2014). "
                            "Use comma, period, colon, or regular hyphen instead.",
                        )
                        break

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all checks pass, False otherwise.
        """
        try:
            pr = self._get_pr()
            commits = self._get_commits()
            files = self._get_files()

            self.check_base_branch_freshness(pr)
            self.check_commit_authorship(commits)
            self.check_commit_messages(commits)
            self.check_pr_title(pr)
            self.check_pr_body(pr)
            self.check_pr_template(pr)
            self.check_changed_files(files)

            return len(self.errors) == 0

        except requests.RequestException as e:
            print(f"Error communicating with GitHub API: {e}", file=sys.stderr)
            return False

    def report(self) -> None:
        """Print validation report."""
        if not self.errors:
            print("✓ All PR guardrails passed")
            return

        print(f"✗ PR guardrails failed with {len(self.errors)} error(s):\n")
        for i, error in enumerate(self.errors, 1):
            print(f"{i}. [{error.check}] {error.message}\n")

        print("\nFix these issues and push again.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate pull request against repository guardrails"
    )
    parser.add_argument("--repo", required=True, help="Repository (owner/name)")
    parser.add_argument("--pr-number", type=int, required=True, help="Pull request number")
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub API token (or set GITHUB_TOKEN env var)",
    )

    args = parser.parse_args()

    if not args.github_token:
        print(
            "Error: GitHub token required. Set GITHUB_TOKEN env var or use --github-token",
            file=sys.stderr,
        )
        return 1

    guard = PRGuard(
        github_token=args.github_token,
        repo=args.repo,
        pr_number=args.pr_number,
    )

    if guard.validate():
        guard.report()
        return 0
    else:
        guard.report()
        return 1


if __name__ == "__main__":
    sys.exit(main())
