FILE_PATH="ml-100k"

# -t: testonly
# -p: prediction output
awk -F"\t" '{printf "%d |u %d |i %d\n", $3,$1,$2}' < $FILE_PATH/ua.test | \
    poetry run python -m vowpalwabbit /dev/stdin \
    -i movielens.reg \
    -p p_out
