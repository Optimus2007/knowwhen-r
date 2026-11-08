#!/bin/sh
# Run before ejecting the project drive so the APFS image closes cleanly.
hdiutil detach /Volumes/KWenv
