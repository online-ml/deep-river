pyreverse --no-standalone --only-classnames --source-roots .. -o dot deep_river
dot -Tpng classes.dot > classes.png
dot -Tpng packages.dot > packages.png
docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara