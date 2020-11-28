#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"

#define DATATIME        100

using namespace std;

enum OPTION {
    ONEHOT,
    //CBOW
};


void MakeOneHotVector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}


template<typename DTYPE>
class TextDataset : public Dataset<DTYPE> {
private:

    char* vocab ;
    char* TextData;

    int vocab_size;
    int text_length;

    Tensor<DTYPE>* m_input;
    Tensor<DTYPE>* m_label;

    OPTION option;

    int VOCAB_LENGTH;

    //dataloader랑 같이 하기
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;

    int m_dimOfInput;
    int m_dimOfLabel;

public:
    TextDataset(string File_Path, int vocab_length, OPTION pOption) {
        vocab = NULL;
        TextData = NULL;

        vocab_size = 0;
        text_length = 0;

        m_input = NULL;
        m_label = NULL;

        option = pOption;

        VOCAB_LENGTH = vocab_length;

        m_aaInput = NULL;
        m_aaLabel = NULL;

        m_numOfInput = 0;

        m_dimOfInput = 0;
        m_dimOfLabel = 0;


        Alloc(File_Path);
    }

    virtual ~TextDataset() {
        Delete();
    }

    void                                  Alloc(string File_Path);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    void                                  MakeInputData();
    void                                  MakeLabelData();

    int                                   char2index(char c);

    char                                  index2char(int index);

    Tensor<DTYPE>*                        GetInputData();

    Tensor<DTYPE>*                        GetLabelData();

    int                                   GetTextLength();

    int                                   GetVocabSize();

    char*                                 GetVocab();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void TextDataset<DTYPE>::Alloc(string File_Path) {

    vocab = new char[VOCAB_LENGTH];

    //File_Reader
    FileReader(File_Path);

    m_dimOfInput = DATATIME;

    m_numOfInput = text_length/DATATIME;         //drop remainder
    m_aaInput = new DTYPE *[m_numOfInput];
    m_aaLabel = new DTYPE *[m_numOfInput];

    //make_vocab
    MakeVocab();

    m_dimOfLabel = vocab_size;

    //make_Input_data
    MakeInputData();

    //make_label_data
    MakeLabelData();
}


template<typename DTYPE> void TextDataset<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void TextDataset<DTYPE>::FileReader(string pFile_Path) {
    ifstream fin;
    fin.open(pFile_Path);

    if(fin.is_open()){

      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.seekg(0, ios::beg);

      TextData = new char[text_length];

      fin.read(TextData, text_length);

    }
    fin.close();
}

template<typename DTYPE> void TextDataset<DTYPE>::MakeVocab(){

    int flag = 0;
    for(int i=0; i<text_length; i++){

        flag = 0;
        vocab_size = (int)strlen(vocab);

        for(int j=0; j<vocab_size; j++){
            if(vocab[j]==TextData[i])
              flag = 1;
            }

        if(flag==0){
          vocab[vocab_size] = TextData[i];
        }
    }

    vocab_size = (int)strlen(vocab)+1;
    sort(vocab, vocab+vocab_size-1);


}

template<typename DTYPE> void TextDataset<DTYPE>::MakeInputData(){

    int index=0;

    for (int i = 0; i < m_numOfInput; i++) {

        m_aaInput[i] = new DTYPE[m_dimOfInput];

        for(int j=0; j<DATATIME; j++){
            m_aaInput[i][j] = char2index(TextData[index]);
            index ++;
        }
    }

}

template<typename DTYPE> void TextDataset<DTYPE>::MakeLabelData(){

/*
    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        m_label = new Tensor<float>(text_length, 1, 1, 1, vocab_size);

        for(int i=0; i<text_length; i++){

            //마지막 data
            if(i==text_length-1){
                MakeOneHotVector(onehotvector, vocab_size, vocab_size-1);
                for(int j=0; j<vocab_size; j++){
                    (*m_label)[Index5D(m_label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
              }
              continue;
            }

            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i+1]));
            for(int j=0; j<vocab_size; j++){
                (*m_label)[Index5D(m_label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }
*/

    int index=1;

    for (int i = 0; i < m_numOfInput; i++) {

        m_aaLabel[i] = new DTYPE[m_dimOfInput];

        for(int j=0; j<DATATIME; j++){
              if( (i+1)*index == text_length){
                  m_aaLabel[i][j] = vocab_size-1;
              } else{
                m_aaLabel[i][j] = char2index(TextData[index]);
                index ++;
            }
        }
    }



}

template<typename DTYPE> int TextDataset<DTYPE>::char2index(char c){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==c)
          return index;
    }
    return -1;
}

template<typename DTYPE> char TextDataset<DTYPE>::index2char(int index){

    if(index == vocab_size-1)
        return 'E';
    else
        return vocab[index];
}

template<typename DTYPE> char* TextDataset<DTYPE>::GetVocab(){

    return vocab;
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset<DTYPE>::GetInputData(){

    return m_input;
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset<DTYPE>::GetLabelData(){
    return m_label;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetVocabSize(){
    return vocab_size;
}


template<typename DTYPE> std::vector<Tensor<DTYPE> *> *TextDataset<DTYPE>::GetData(int idx) {

      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(DATATIME, 1, 1, 1, 1);
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(DATATIME, 1, 1, 1, m_dimOfLabel);

      for (int i = 0; i < DATATIME; i++) {
          (*input)[i] = m_aaInput[idx][i];
      }

      int* onehotvector = new int[vocab_size];
      for (int i=0; i<DATATIME; i++){
          MakeOneHotVector(onehotvector, vocab_size, m_aaLabel[idx][i]);
          for(int j=0; j<vocab_size; j++){
              (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
          }
      }


      result->push_back(input);
      result->push_back(label);

      return result;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetLength() {
        return m_numOfInput;
}