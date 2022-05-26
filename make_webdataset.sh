mkdir -p data/webdataset/sg2-ffhq-1024-w3

for i in {00000..00099}
do
    tar --sort=name -cf data/webdataset/sg2-ffhq-1024-w3/$i.tar data/sg2-ffhq-1024-w3/$i
done
