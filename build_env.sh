# build image 
docker build -t fms5_envimg .

# run the image as a container
docker run -d -it --name fms5_cntnr -p 4208:4208 -v `pwd`/test_out:/local_vol/test_out fms5_envimg  bash
