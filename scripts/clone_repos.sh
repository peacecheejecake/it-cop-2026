#!/bin/bash
# Bare-clone public Apache repositories into the shared cache (read-only afterwards).
# jd4j/: the 21 JIT-Defects4J projects (full diffs of labeled commits).
# cpt/:  Java Apache projects disjoint from JD4J (unlabeled diff pretraining corpus; no SZZ fix-commit leakage into JD4J).
set -u
ROOT=$(cd "$(dirname "$0")/../../.cache/git" && pwd)
LOG=$(cd "$(dirname "$0")/../logs" && pwd)
JD4J="ant-ivy commons-bcel commons-beanutils commons-codec commons-collections commons-compress commons-configuration commons-dbcp commons-digester commons-io commons-jcs commons-lang commons-math commons-net commons-scxml commons-validator commons-vfs giraph gora opennlp parquet-mr"
CPT="zookeeper zeppelin activemq kafka cassandra groovy"
clone() { # group repo
  local dst="$ROOT/$1/$2.git"
  if [ -d "$dst" ]; then echo "$(date -u +%T) skip $1/$2 (exists)"; return; fi
  if git clone --bare --quiet "https://github.com/apache/$2.git" "$dst.tmp" > "$LOG/clone-$1-$2.log" 2>&1; then
    mv "$dst.tmp" "$dst"; echo "$(date -u +%T) ok $1/$2 $(du -sh "$dst" | cut -f1) $(git -C "$dst" rev-parse HEAD)"
  else
    rm -rf "$dst.tmp"; echo "$(date -u +%T) FAIL $1/$2 (see $LOG/clone-$1-$2.log)"
  fi
}
for r in $JD4J; do clone jd4j "$r"; done
for r in $CPT; do clone cpt "$r"; done
echo "$(date -u +%T) DONE"
