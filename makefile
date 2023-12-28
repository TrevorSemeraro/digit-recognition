# # Compiler flags
CC = g++
SRC = main.cpp

CFLAGS = -Wall -std=c++11

# apply: apply.o network.o lib.o layer.o learn.o save.o
# 	${CC} ${CFLAGS} -o $@ apply.o network.o lib.o layer.o learn.o save.o

# apply.o: apply.cpp
# 	${CC} ${CFLAGS} -o $@ -c apply.cpp

main : main.o network.o lib.o layer.o learn.o save.o
	${CC} ${CFLAGS} -o $@ main.o network.o lib.o layer.o learn.o save.o

main.o : main.cpp
	${CC} ${CFLAGS} -o $@ -c main.cpp

network.o : network.cpp network.h
	${CC} ${CFLAGS} -o $@ -c network.cpp

lib.o : lib.cpp lib.h
	${CC} ${CFLAGS} -o $@ -c lib.cpp

layer.o : layer.cpp layer.h
	${CC} ${CFLAGS} -o $@ -c layer.cpp

learn.o : learn.cpp learn.h
	${CC} ${CFLAGS} -o $@ -c learn.cpp

save.o : save.cpp save.h
	${CC} ${CFLAGS} -o $@ -c save.cpp

clean:
	-del -fR *.o main.exe