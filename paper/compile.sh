#!/bin/bash
pyreverse --no-standalone --only-classnames --source-roots .. -o dot deep_river
grep -v -E "Anomaly[a-zA-Z]*Scaler" classes.dot > tmpfile.dot && mv tmpfile.dot classes.dot
grep -v -E "Anomaly[a-zA-Z]*Scaler" packages.dot > tmpfile.dot && mv tmpfile.dot packages.dot

dot -Tpng classes.dot > classes.png
dot -Tpng packages.dot > packages.png
docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara