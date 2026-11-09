# ADR-008: Weekly track record

**Date**: 2026-11-09
**Status**: Accepted

## Context

The page publishes one slate per NFL week. ADR-006 freezes that slate. ADR-007 adds the estimated margin and total. Final scores are overlaid later. The orphan branch `published-slates` keeps the file that was public, and `history/` keeps every earlier copy.

A track record has to use those files. Scoring the week again, or reading a rebuilt file as if it were the original, would not be the slate readers saw before kickoff. The sample is small. Week 9 of 2026 is the first archived week. The page has to say that, and it has to stay an open experiment.

## Decision

`python -m endzone.site.track_record` reads the archive and writes `site/data/track_record.json`. Pages and Results run it after the score overlay and before the site upload. The file is derived. It is not a week slate, it is not committed on `main`, and it is not pushed to `published-slates`.

The command does not modify, overwrite, or shrink a week file, `latest.json`, or anything in the archive. A path whose name is a week file, or a path inside the archive or `predictions/`, is refused.

### Which file counts

For each `game_id`, probabilities and point estimates come from the earliest archived snapshot that contains that game. History files are older than the tip. The tip is older than a rescore that has not been archived yet.

Final scores come from the newest copy. During a deploy that copy is the week file in `site/data` after the results overlay. If that file has no row, the archive tip is next, then older history.

A game with no archived snapshot uses the week file this build is about to publish. That is a first publish, not a rescore. The document counts those games separately.

An example slate (`sample: true`) is ignored.

### Pick hits

The pick is the same one the game card draws. The home win probability is rounded to a percent, half away from zero, matching the page. 50 is a tie pick. Above 50 is the home team. Below 50 is the away team.

A hit is a finished game whose pick equals the result. A final tie is a hit only when the pick is also a tie. Any other pick on a tie is a miss. The page shows hits out of finished games, and it shows the tie count beside that.

A finished game with no usable probability is not a hit and not a miss. The count is `pick_unavailable`.

### Log loss

Log loss is the mean binary log loss, natural log, on finished games that have a winner and a market probability. The model and the market are scored on that same set. A game with no market price is left out of both, and the page says how many decided games had a price.

A final tie is not a 0 or 1 label, so ties are left out of both log losses.

Probabilities are clipped to `[1e-6, 1 - 1e-6]` before the log. A stored 0 or 1 stays finite. A value outside `[0, 1]`, or a missing value, drops the game from both log losses.

Lower log loss is closer to the results. The table does not declare a winner.

### Score error

Score error is two mean absolute errors, in points, on finished games that published both estimates. Ties are included. Unfinished games are not.

- Margin error: `estimated_margin` against home score minus away score.
- Total error: `estimated_total` against home score plus away score.

Those are the published model outputs. The page rounds team points for display only. Team-score error would split the same two numbers again, so the table shows margin and total, the same pair as the Model accuracy block. Games with a null estimate are left out, and they are not filled in from a later fit.

### Unfinished games and partial weeks

`status` other than `final`, or a final without two non-negative integer scores, is unfinished. Unfinished games stay out of hits, log loss, and score error. They still count in the week's game total, so the page can show "14 of 15".

A week with no finals is a row, with rates left blank. A week with some finals is "in progress" and uses only the finals. An empty archive is an empty state, not a row of zeros.

### Season to date

Each season gets one summary row. Counts are sums. Log loss and the two errors are means over the games, not the mean of the weekly means.

The summary is labeled when the sample is small:

- `no_finals` when no week has a final.
- `one_week` when exactly one week has a final.
- `small` when more than one week has a final and fewer than 48 decided games are in the pool. 48 is about three full weeks.
- No label after that.

The page says "small sample" for `one_week` and `small`. It does not treat one week as a season result.

### When the archive cannot be read

If the archive directory is missing, or the `published-slates` branch cannot be read, the command writes `available: false` and `reason: archive_unreachable`. There are no week rows and no rates. The page says the build does not show a track record. It does not fall back to whatever week files happen to be on disk, because those files may be a rescore.

A week file that cannot be parsed, or that names a different season or week than its path, fails the command. A stale `track_record.json` is deleted. The deploy does not publish a partial record. An unreachable archive is the omit case. A corrupt archive is the fail case.

## Consequences

The Pages push, the Tuesday score, and the Results overlay all write this file before upload. Asset URLs in `site/index.html` get a content hash at the same time, because Pages caches files for 10 minutes and the HTML otherwise keeps the old CSS and JavaScript name. The track record request also sends a unique query, for the same cache.

The 0.003 rule in ADR-005 is unchanged. This record is a description of published weeks, not a reason to move a threshold.
