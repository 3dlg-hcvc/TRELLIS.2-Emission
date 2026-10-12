#!/bin/bash
# The encode stage of script/internal/alliance/lightgenbench_v1.sh (usage and env there); one path per stage for hedge.sh.
STAGE=encode exec bash "$(dirname "$0")/lightgenbench_v1.sh" "$@"
