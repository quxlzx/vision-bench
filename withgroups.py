#!/usr/bin/env python3
"""Run a command with extra supplementary groups (render/video/docker)
without needing a login restart. Usage: withgroups.py <cmd...>"""
import grp, os, sys
want = ['render', 'video', 'docker']
gids = list(os.getgroups())
for g in want:
    try:
        gid = grp.getgrnam(g).gr_gid
        if gid not in gids:
            gids.append(gid)
    except KeyError:
        pass
os.setgroups(gids)
print('groups:', sorted(os.getgroups()), file=sys.stderr)
os.execvp(sys.argv[1], sys.argv[1:])
