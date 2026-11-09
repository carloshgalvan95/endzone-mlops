"""Static-site data updates that do not fit a model.

``python -m endzone.site.results`` overlays final nflverse scores onto week
files that scoring already published. ``python -m endzone.site.archive``
restores those files from the ``published-slates`` branch, then from the
live site, and writes the files a deploy is about to publish back to that
branch. ``python -m endzone.site.track_record`` reads that archive and
writes ``site/data/track_record.json``. It does not edit a week file.
"""
