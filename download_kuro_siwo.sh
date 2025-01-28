#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi

root_folder_path=$1

full_path="$root_folder_path/KuroSiwo"

mkdir $full_path

cat_url="https://www.dropbox.com/scl/fi/wu6nvj73cz4h7k3gxpzx6/catalogue.gpkg?rlkey=hsij2o0k60r2n0z6z4d2ngww9&st=0zjqhzgx&dl=0"
batch_1_url="https://www.dropbox.com/scl/fi/19mm9v5pnd5yor8b15alj/00.tar.gz?rlkey=f7qrqgv7h7z9j6r595xz1720e&st=k4ptwanm&dl=0"
batch_2_url="https://www.dropbox.com/scl/fi/pjlgcqc3fm8lx97vdfjyn/01.tar.gz?rlkey=bvue0u3jgovc3qewkh464uyog&st=qfsa601d&dl=0"
batch_3_url="https://www.dropbox.com/scl/fi/7tt843025s8hqi00xpofx/02.tar.gz?rlkey=8vbfh3qc2h7zmzjem0c14lljh&st=xda2zf5j&dl=0"
batch_4_url="https://www.dropbox.com/scl/fi/v7997kv11cc8ptj1vstim/03.tar.gz?rlkey=yj0qspgmlmohaf728a94curqf&st=6qmn63y7&dl=0"
batch_5_url="https://www.dropbox.com/scl/fi/2o3cxwxu79phijxylm99h/04.tar.gz?rlkey=3mcmjrzbf8vmxwl8aacou7vvr&st=yh0e4p72&dl=0"
batch_6_url="https://www.dropbox.com/scl/fi/q1jy4ep4j6bj38dva2es3/05.tar.gz?rlkey=g46498pgox03it080p83xhx8g&st=haxheepl&dl=0"
batch_7_url="https://www.dropbox.com/scl/fi/843fba3poe6nu67og5n6x/06.tar.gz?rlkey=bnxhf9zj0y8mtvjv7k683ywaa&st=p4pg93ko&dl=0"
batch_8_url="https://www.dropbox.com/scl/fi/89kogi7nisinbfa3z2wze/07.tar.gz?rlkey=byduc9q5cndwjphq4r1727xrl&st=a2sgt9dp&dl=0"
batch_9_url="https://www.dropbox.com/scl/fi/sctg1ybqxtzz23018htwx/08.tar.gz?rlkey=6hj887jxezjq5gibqz4zhh7gd&st=9gcgiwdu&dl=0"
batch_10_url="https://www.dropbox.com/scl/fi/425wrfqqc7sy8pbdnkgzw/09.tar.gz?rlkey=ziejzpo2ddgnygpbd1emn1nir&st=irjm2j98&dl=0"
batch_11_url="https://www.dropbox.com/scl/fi/rwc2yvd0g070qicg300de/10.tar.gz?rlkey=12i17si2kzjpey4vgig9pjnv7&st=p2v45gue&dl=0"

urls=($cat_url $batch_1_url $batch_2_url $batch_3_url $batch_4_url $batch_5_url $batch_6_url $batch_7_url $batch_8_url $batch_9_url $batch_10_url $batch_11_url)

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