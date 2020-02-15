FILE_NAME="dataset.zip"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&confirm=YiKU&id=1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
