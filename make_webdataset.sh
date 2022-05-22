mkdir -p data/webdataset/sg2-ffhq-1024-clip

for i in {00000..00099}
do
    tar --sort=name -cf data/webdataset/sg2-ffhq-1024-clip/$i.tar data//sg2-ffhq-1024-clip/$i
done
