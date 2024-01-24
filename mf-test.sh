FILE_PATH="ml-100k"

# -t: testonly
# -p: prediction output
awk -F"\t" '{printf "%d |u %d |i %d\n", $3,$1,$2}' < $FILE_PATH/ua.test > test.data

poetry run python -m vowpalwabbit \
-d test.data \
-i movielens.reg \
-p p_out

rm test.data
