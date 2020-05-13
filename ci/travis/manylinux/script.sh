#!/bin/sh

set -e

. $(dirname $(dirname $0))/functions.sh

# --- Test -------------------------------------------------------------------

case $TRAVIS_PYTHON_VERSION in
  pypy3) TAG=pp36-pypy3_*;;
  *)     TAG=cp$(echo $TRAVIS_PYTHON_VERSION | sed 's/\.//');;
esac

log Running test with $TAG
docker exec -it manylinux sh /io/ci/travis/manylinux/_script.sh $TAG
