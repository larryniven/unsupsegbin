CXXFLAGS += -std=c++11 -I .. -L ../speech -L ../nn -L ../autodiff -L ../opt -L ../la -L ../ebt -L ../fst -L ../unsupseg

bin = \
    random-seg \
    conv-embed \
    conv-embed-kmeans \
    conv-embed-kmeans-predict \
    conv-kmeans-learn \
    conv-kmeans-predict \
    dtw \
    dtw-embed \
    dtw-embed-kmeans \
    dtw-embed-kmeans-predict \
    dtw-lstm-learn \
    dtw-lstm-predict \
    rsg-unsup-learn \
    rsg-unsup-predict

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

random-seg: random-seg.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lla -lebt -lblas

conv-embed: conv-embed.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

conv-embed-kmeans: conv-embed-kmeans.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

conv-embed-kmeans-predict: conv-embed-kmeans-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

conv-kmeans-learn: conv-kmeans-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lnn -lopt -lautodiff -lla -lebt -lblas

conv-kmeans-predict: conv-kmeans-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lnn -lopt -lautodiff -lla -lebt -lblas

dtw: dtw.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

dtw-embed: dtw-embed.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

dtw-embed-kmeans: dtw-embed-kmeans.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

dtw-embed-kmeans-predict: dtw-embed-kmeans-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lspeech -lla -lebt -lblas

dtw-lstm-learn: dtw-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

dtw-lstm-predict: dtw-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lunsupseg -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rsg-unsup-learn: rsg-unsup-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rsg-unsup-predict: rsg-unsup-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

