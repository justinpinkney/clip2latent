mkdir -p data/webdataset/sg3-lhq-256-clip

for i in {00000..00099}
do
    tar --sort=name -cf data/webdataset/sg3-lhq-256-clip/$i.tar data/sg3-lhq-256-clip/$i
done
