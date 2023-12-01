#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi

root_folder_path=$1

full_path="$root_folder_path/KuroSiwo"

mkdir $full_path

cat_url="https://www.dropbox.com/scl/fi/vl9iea5k9fvq35oaybsy2/catalogue.gpkg?rlkey=5ekxjfj9ynor7gk9rom7st5kd&dl=0"
batch_1_url="https://www.dropbox.com/scl/fi/vcmzontrqlywkb6827xr4/01.tar.gz?rlkey=p08coyujc8jtqc9omp8pee4u9&dl=0"
batch_2_url="https://www.dropbox.com/scl/fi/9eiskq6bvwvtkcgar7fau/02.tar.gz?rlkey=jjd8uitqa31wmivcirg84q83o&dl=0"
batch_3_url="https://www.dropbox.com/scl/fi/wal95mj2lgrhn5vfv82ru/03.tar.gz?rlkey=fxgfccf16v8bmawhkrlivat3s&dl=0"
batch_4_url="https://www.dropbox.com/scl/fi/r6zfyfkauctuvdgondul0/04.tar.gz?rlkey=zq4rnjljvyph7udidfntgn5p1&dl=0"
batch_5_url="https://www.dropbox.com/scl/fi/rwvumfstxp27k1akv0hv5/05.tar.gz?rlkey=ukysm10tljygxourdkbwzkr2k&dl=0"
batch_6_url="https://www.dropbox.com/scl/fi/uqobj3knewqm5y6a9cax3/06.tar.gz?rlkey=oub1c8905gwuthjlvz6lydib1&dl=0"
batch_7_url="https://www.dropbox.com/scl/fi/7uamc6g165so3mfv9n411/07.tar.gz?rlkey=53x6tmmmh7iqbbzizu64vonco&dl=0"
batch_8_url="https://www.dropbox.com/scl/fi/ejztwj7ib545qsk7kw8ou/08.tar.gz?rlkey=tbl0bk3kzqe77ugs5fucdpkd4&dl=0"
urls=($cat_url $batch_1_url $batch_2_url $batch_3_url $batch_4_url $batch_5_url $batch_6_url $batch_7_url $batch_8_url)

downloaded_files=()
failed_downloads=()
failed_extractions=()

for i in "${!urls[@]}"; do
    file_url="${urls[i]}"
    echo "Downloading $file_url"
    if [ "$i" -eq 0 ]; then
        filename="catalogue.gpkg"
    else
        filename=$i
        filename="${filename}.tar.gz"
    fi
    echo "Storing to $filename"
    wget -c $file_url -O "$full_path/$filename"
    if [ $? -eq 0 ]; then
      echo "$file_url downloaded successfully"
      downloaded_files+=("$full_path/$filename")
    else
      echo "$file_url downloading failed"
      failed_downloads+=("$file_url")
    fi 
done

echo "Begin extraction for files: "
for i in "${!downloaded_files[@]}"; do
    url=${downloaded_files[i]}
    echo "${url}"
done

for i in "${!downloaded_files[@]}"; do
    
    file="${downloaded_files[i]}"
    if [[ $file == *.gpkg ]]; then
        echo "Catalogue file does not need extraction!"
        continue
    fi
    tar -xvf $file --directory=$full_path
    if [ $? -eq 0 ]; then
        echo "$file extracted successfully! Removing tar file!"
        rm $file
    else
        echo "$file extraction failed!"
        failed_extractions+=("$file")
    fi
done

flag=0
if [ ${#failed_downloads[@]} -ne 0 ]; then
    echo "Downloading failed for: "
    for i in "${!failed_downloads[@]}";do
        url=${failed_downloads[i]}
        echo "$url"
    done
    if [ ${#failed_extractions[@]} -ne 0 ]; then
        echo "Extraction failed for: "
        for i in "${!failed_extraction[@]}";do
            file=${failed_extraction[i]}
            echo "$file"
        done
        flag++
    fi
    flag++
fi

if [ ${flag} -eq 0 ]; then
    echo "Kuro Siwo downloading finished successfully!"
fi