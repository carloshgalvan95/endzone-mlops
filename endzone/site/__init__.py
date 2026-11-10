"""Static-site data updates that do not fit a model.

``python -m endzone.site.results`` overlays final nflverse scores onto week
files that scoring already published. ``python -m endzone.site.archive``
restores those files from the ``published-slates`` branch, then from the
live site, and writes the files a deploy is about to publish back to that
branch. ``python -m endzone.site.track_record`` reads that archive and
writes ``site/data/track_record.json``. It does not edit a week file.
``python -m endzone.site.score_gate`` reports whether the target week
file is already present so a Pages run can skip the fit.
``python -m endzone.site.slate_check`` opens one issue when that file
or the previous week's finals are missing. It does not write a branch.
"""
