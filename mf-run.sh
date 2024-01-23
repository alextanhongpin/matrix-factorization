FILE_PATH="ml-100k"

awk -F"\t" '{printf "%d |u %d |i %d\n", $3,$1,$2}' < $FILE_PATH/ua.base | \
    poetry run python -m vowpalwabbit /dev/stdin \
    -b 18 \
    -q ui \
    --rank 10 \
    --l2 0.001 \
    --learning_rate 0.015 \
    --passes 20 \
    --decay_learning_rate 0.97 \
    --power_t 0 \
    -f movielens.reg \
    --cache_file movielens.cache
