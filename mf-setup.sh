FILE_PATH="ml-100k.zip"
URL="http://files.grouplens.org/papers/ml-100k.zip"

if [ -f "$FILE_PATH" ]; then
    echo "File $FILE_PATH exists."
else
    echo "File $FILE_PATH does not exist. Downloading..."
    curl -o $FILE_PATH $URL
fi

unzip -n $FILE_PATH
