mkdir -p ffhq/webdataset

for i in {00000..00001}
do
    tar --sort=name -cf ffhq/webdataset/$i.tar dump/$i
done