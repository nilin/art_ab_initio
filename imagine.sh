mkdir pics/imagination

for content_path in pics/cont_imag/*; do
    contentname=$(basename -- "$content_path")
    contentname="${contentname%.*}"

    for path in pics/inspiration/*; do
        filename=$(basename -- "$path")
        filename="${filename%.*}"
        outpath="pics/imagination/c_${contentname}_i_${filename}.jpg"
        echo $path
        echo $content_path
        echo $outpath

        python style_transfer/styletransfer.py \
            --style_path $path \
            --content_path $content_path \
            --out_path $outpath

        python style_transfer/recolor.py $outpath $content_path $outpath
        python style_transfer/revalue.py $outpath $content_path $outpath
    done
done
