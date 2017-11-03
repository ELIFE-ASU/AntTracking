for folder in ant_img/*; do
	echo $folder
	i=0
	for file in $folder/*.png; do
		mv "$file" $folder/$i.png
		((i++))
	done
done
