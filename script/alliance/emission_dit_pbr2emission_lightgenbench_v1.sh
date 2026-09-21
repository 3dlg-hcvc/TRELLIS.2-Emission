#!/bin/bash
# The dit stage of script/alliance/lightgenbench_v1.sh (usage and env there); one path per stage for hedge.sh.
STAGE=dit exec bash "$(dirname "$0")/lightgenbench_v1.sh" "$@"
