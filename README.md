# Published slates

This branch is the durable copy of the weekly prediction JSON. The Pages
and Results workflows write it. Do not edit it by hand, and do not merge
it into main.

`predictions/<season>-w<week>.json` is the current file for that week.
`latest.json` is the file the page loads. `history/` keeps every earlier
copy. The file name is the UTC time of that archive write, with no colon,
so it is safe on Windows. A later week file does not delete the earlier one.

main does not store these files. See ADR-006 on main.
