sudo docker build -t my-custom-postgres 
sudo docker network create my_network
sudo docker run --name my-custom-postgres-instance --network my_network -p 5432:5432 -d my-custom-postgres
psql -h localhost -p 5432 -U alatar -d mydb
# save docker image
sudo docker save -o my-custom-postgres.tar my-custom-postgres


sudo docker stop my-custom-postgres
sudo docker rm my-custom-postgres
