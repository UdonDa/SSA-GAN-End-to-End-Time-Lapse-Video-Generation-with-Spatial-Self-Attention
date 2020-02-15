FILE_ID="1akRfnKxaBrkLLACNc70QvVJr3DseSfK6"
FILE_NAME="md_s1_030.pth"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&confirm=YiKU&id=1akRfnKxaBrkLLACNc70QvVJr3DseSfK6" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}


FILE_ID="15-4bWx1esI0OLiW_gDc683zc7drWG45s"
FILE_NAME="ms_s2_067.pth"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&confirm=YiKU&id=15-4bWx1esI0OLiW_gDc683zc7drWG45s" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
