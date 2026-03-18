CC      ?= cc
CFLAGS  ?= -O2 -march=native -Wall -Wextra -Wpedantic
LDLIBS  := -lm -lpthread

.PHONY: all clean test

all: test_embedding

embedding.o: embedding.c embedding.h
	$(CC) $(CFLAGS) -c -o $@ embedding.c

test_embedding: test_embedding.c embedding.o embedding.h
	$(CC) $(CFLAGS) -o $@ test_embedding.c embedding.o $(LDLIBS)

test: test_embedding
	./test_embedding models/embeddinggemma

clean:
	$(RM) embedding.o test_embedding
