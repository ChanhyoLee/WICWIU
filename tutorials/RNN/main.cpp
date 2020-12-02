#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "CharTextGeneratorDataset.hpp"
#include "net/my_RNN.hpp"
//#include "TextDataset.hpp"

using namespace std;

#define EMBEDDIM 64
#define TIME_MAIN 250
#define BATCH_MAIN 5
#define EPOCH 5
#define MAX_TRAIN_ITERATION (1000 / BATCH_MAIN)
#define MAX_TEST_ITERATION (1000 / BATCH_MAIN)
#define GPUID 2

int main(int argc, char const *argv[]) {
    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    CharTextGeneratorDataset<float> *dataset = new CharTextGeneratorDataset<float>("Data/test.txt", "?.!;");  //char단위

    std::cout << "데이터셋 생성 완료" << std::endl;

    DataLoader<float> *train_dataloader = new DataLoader<float>(dataset, BATCH_MAIN, TRUE, 20, FALSE);

    std::cout << "데이터로더 생성 완료" << std::endl;

    int text_length = dataset->GetTextLength();
    int vocab_size = dataset->GetVocabSize();

    std::cout << "파일 길이 : " << text_length << " vocab 길이 : " << vocab_size << '\n';

    //Tensorholder<float> *x_holder = new Tensorholder<float>(Text_length, BATCH_MAIN, 1, 1, vocab_size, "x");       //Basis RNN
    Tensorholder<float> *x_holder = new Tensorholder<float>(TIME_MAIN, BATCH_MAIN, 1, 1, 1, "x");  //char generation
    Tensorholder<float> *label_holder = new Tensorholder<float>(TIME_MAIN, BATCH_MAIN, 1, 1, vocab_size, "label");
    std::cout << "텐서홀더 생성 완료" << '\n';

    NeuralNetwork<float> *net = new my_RNN(x_holder, label_holder, vocab_size, EMBEDDIM);

#ifdef __CUDNN__
    std::cout << "GPU환경에서 실행중 입니다." << '\n';
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

#ifdef __CUDNN__
    x->SetDeviceGPU(GPUID);
    label->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    std::cout << '\n';
    net->PrintGraphInformation();

    float best_acc = 0;
    int   epoch = 0;

    for (int i = epoch + 1; i < EPOCH; i++) {
        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        // ============================== Train ===============================
        std::cout << "Start Train" << '\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {
            std::vector<Tensor<float> *> *temp = train_dataloader->GetDataFromGlobalBuffer();

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *l_t = (*temp)[1];
            delete temp;

            net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            net->BPTT(TIME_MAIN);

            train_accuracy = net->GetAccuracy(vocab_size);
            train_avg_loss = net->GetLoss();

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss,
                   train_accuracy);

            if (j % 300 == 0)
                net->GetCharResult(dataset->GetVocab());
            fflush(stdout);
        }

        endTime = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "Start Test" << '\n';

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {
            int startIndex = 0;

            std::cout << '\n'
                      << "F로 시작" << '\n';
            startIndex = dataset->GetpChar2Index()->at('F');
            net->GenerateSentence(TIME_MAIN, dataset->GetVocab(), startIndex, vocab_size);

            //fflush(stdout);
            std::cout << "\n\n";
        }

    }  // 여기까지가 epoc for문

    delete net;

    return 0;
}
