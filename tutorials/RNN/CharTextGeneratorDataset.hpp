#ifndef __TEXT_HPP__
#define __TEXT_HPP__

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../WICWIU_src/DataLoader.hpp"
#include "../../WICWIU_src/Tensor.hpp"

#define DATATIME 250

void MakeOneHotVector(int* onehotvector, int vocab_size, int index) {
    for (int i = 0; i < vocab_size; i++) {
        if (i == index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}

using namespace std;

template <typename DTYPE> class CharTextGeneratorDataset : public Dataset<DTYPE> {
private:
    string          path;
    string          delimiters;
    char*           m_pTextData;
    char*           m_aVocab;
    int             m_TextLength;
    int             m_LineSize;
    int             m_MaxSentenceLength;
    map<int, char>* m_pIndex2Char;
    map<char, int>* m_pChar2Frequency;
    map<char, int>* m_pChar2Index;
    int             n_Characters;
    vector<string>* m_pSentences;
    vector<int>*    m_pSentenceSizes;
    DTYPE**         m_aaInput;
    DTYPE**         m_aaLabel;
    int             m_numOfInput;
    int             m_dimOfLabel;

public:
    CharTextGeneratorDataset(string _path, string _delimiters);
    ~CharTextGeneratorDataset();
    void                                 Alloc();
    void                                 Delete();
    void                                 ReadFile();
    void                                 ReadLine();
    void                                 BuildVocab();
    void                                 AddChar(char token);
    vector<string>*                      GetpSentences();
    vector<int>*                         GetpSentenceSizes();
    map<char, int>*                      GetpChar2Index();
    char*                                GetVocab();
    void                                 MakeInputData();
    void                                 MakeLabelData();
    virtual std::vector<Tensor<DTYPE>*>* GetData(int idx);
    char*                                SubString(char* dst, const char* src, size_t n);
    int                                  GetLength();
    int                                  GetVocabSize();
    int                                  GetTextLength();
};

template <typename DTYPE> CharTextGeneratorDataset<DTYPE>::CharTextGeneratorDataset(string _path, string _delimiters) {
    path = _path;
    delimiters = _delimiters;

    m_pTextData = NULL;
    m_TextLength = 0;
    m_LineSize = 0;

    m_pIndex2Char = NULL;
    m_pChar2Frequency = NULL;
    m_pChar2Index = NULL;
    n_Characters = 0;

    m_aaInput = NULL;
    m_aaLabel = NULL;
    m_numOfInput = 0;

    Alloc();
}

template <typename DTYPE> CharTextGeneratorDataset<DTYPE>::~CharTextGeneratorDataset() {
    Delete();
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::Delete() {
    if (m_aaInput) {
        delete[] m_aaInput;
        m_aaInput = NULL;
    }

    if (m_aaLabel) {
        delete[] m_aaLabel;
        m_aaLabel = NULL;
    }

    if (m_pTextData) {
        delete[] m_pTextData;
        m_pTextData = NULL;
    }

    if (m_pIndex2Char) {
        delete m_pIndex2Char;
        m_pIndex2Char = NULL;
    }

    if (m_pChar2Frequency) {
        delete m_pChar2Frequency;
        m_pChar2Frequency = NULL;
    }

    if (m_pChar2Index != NULL) {
        delete m_pChar2Index;
        m_pChar2Index = NULL;
    }

    if (m_pSentences != NULL) {
        delete m_pSentences;
        m_pSentences = NULL;
    }
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::Alloc() {
    m_pIndex2Char = new map<int, char>();
    m_pChar2Frequency = new map<char, int>();
    m_pChar2Index = new map<char, int>();

    m_pSentences = new vector<string>;
    m_pSentenceSizes = new vector<int>;

    ReadFile();

    ReadLine();

    m_aVocab = new char[n_Characters];
    BuildVocab();

    m_numOfInput = m_pSentences->size();
    m_aaInput = new DTYPE*[m_numOfInput];
    m_aaLabel = new DTYPE*[m_numOfInput];
    m_dimOfLabel = n_Characters;

    MakeInputData();
    MakeLabelData();
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::BuildVocab() {
    for (int i = 0; i < n_Characters; i++) {
        m_aVocab[i] = m_pIndex2Char->at(i);
    }
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::ReadFile() {
    cout << "<<<<<<<<<<<<<<<<  FileReader  >>>>>>>>>>>>>>>>>>>>" << endl;
    this->path = path;
    cout << this->path << endl;
    ifstream fin;
    fin.open(path);

    if (fin.is_open()) {
        fin.seekg(0, ios::end);
        m_TextLength = fin.tellg();
        fin.tellg();
        fin.seekg(0, ios::beg);

        m_pTextData = new char[m_TextLength];
        //파일 읽기
        fin.read(m_pTextData, m_TextLength);

        m_TextLength = strlen(m_pTextData);
        fin.close();
    }

    else {
        cout << "ERROR : CANNOT OPEN FILE" << endl;
        cout << "PATH : " << path << endl;  //파일 없거나 안열릴시
        exit(-1);
    }
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::ReadLine() {
    char   token;
    int    start = 0;
    int    length = 0;
    string temp;
    size_t p;

    for (int i = 0; i < m_TextLength - 1; i++) {
        // printf("%d th character: %c\n", i, m_pTextData[i]);
        token = m_pTextData[i];

        if (m_pChar2Index->find(token) == m_pChar2Index->end())  // Char 검색 및 추가
            AddChar(token);
        else
            m_pChar2Frequency->at(token)++;

        p = delimiters.find(m_pTextData[i]);  // delimiters에 추가 및 sentence segmentation
        if (p == string::npos)
            continue;
        else {
            char temp_sentence[i - start + 2];
            SubString(temp_sentence, m_pTextData + start, i - start + 2);
            string str(temp_sentence);
            m_pSentences->push_back(str);
            m_pSentenceSizes->push_back(str.size());
            start = i + 2;
        }
    }

    m_LineSize = m_pSentences->size();
    int max_len = 0;
    for (int i = 0; i < m_LineSize; i++) {
        cout << i << "th length: " << m_pSentenceSizes->at(i) << endl;
        if (m_pSentenceSizes->at(i) > max_len) max_len = m_pSentenceSizes->at(i);
    }
    cout << "최대 time: " << max_len << endl;
    m_MaxSentenceLength = max_len;
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::AddChar(char token) {
    m_pChar2Index->insert(make_pair(token, n_Characters));
    m_pChar2Frequency->insert(make_pair(token, 1));
    m_pIndex2Char->insert(make_pair(n_Characters, token));
    n_Characters++;
}

template <typename DTYPE> vector<string>* CharTextGeneratorDataset<DTYPE>::GetpSentences() {
    return m_pSentences;
}
template <typename DTYPE> vector<int>* CharTextGeneratorDataset<DTYPE>::GetpSentenceSizes() {
    return m_pSentenceSizes;
}
template <typename DTYPE> char* CharTextGeneratorDataset<DTYPE>::GetVocab() {
    return m_aVocab;
}

template <typename DTYPE> char* CharTextGeneratorDataset<DTYPE>::SubString(char* dst, const char* src, size_t n) {
    dst[0] = '\0';
    return strncat(dst, src, n - 1);
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::MakeInputData() {
    cout << "MakeInputData 호출" << endl;

    int present_index = 0;
    for (int i = 0; i < m_numOfInput; i++) {
        m_aaInput[i] = new DTYPE[m_pSentenceSizes->at(i)];
        for (int j = 0; j < m_pSentenceSizes->at(i); j++) {
            m_aaInput[i][j] = (DTYPE)(m_pChar2Index->at(m_pTextData[present_index]));
            present_index++;
        }
    }
}

template <typename DTYPE> void CharTextGeneratorDataset<DTYPE>::MakeLabelData() {
    cout << "MakeLabelData 호출" << endl;
    int present_index = 1;
    for (int i = 0; i < m_numOfInput; i++) {
        m_aaLabel[i] = new DTYPE[m_pSentenceSizes->at(i)];
        for (int j = 0; j < m_pSentenceSizes->at(i); j++) {
            m_aaLabel[i][j] = (DTYPE)(m_pChar2Index->at(m_pTextData[present_index]));
            present_index++;
        }
    }
}

template <typename DTYPE> std::vector<Tensor<DTYPE>*>* CharTextGeneratorDataset<DTYPE>::GetData(int idx) {
    std::vector<Tensor<DTYPE>*>* result = new std::vector<Tensor<DTYPE>*>(0, NULL);
    Tensor<DTYPE>*               input = Tensor<DTYPE>::Zeros(m_MaxSentenceLength, 1, 1, 1, 1);
    Tensor<DTYPE>*               label = Tensor<DTYPE>::Zeros(m_MaxSentenceLength, 1, 1, 1, m_dimOfLabel);

    for (int i = 0; i < m_pSentenceSizes->at(idx); i++) {
        (*input)[i] = m_aaInput[idx][i];
    }

    int* onehotvector = new int[n_Characters];
    for (int i = 0; i < m_pSentenceSizes->at(idx); i++) {
        if (i > m_pSentenceSizes->at(idx)) break;
        MakeOneHotVector(onehotvector, n_Characters, m_aaLabel[idx][i]);
        for (int j = 0; j < n_Characters; j++) {
            (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
        }
    }

    result->push_back(input);
    result->push_back(label);
    return result;
}

template <typename DTYPE> int CharTextGeneratorDataset<DTYPE>::GetLength() {
    return m_numOfInput;
}
template <typename DTYPE> int CharTextGeneratorDataset<DTYPE>::GetTextLength() {
    return m_TextLength;
}
template <typename DTYPE> int CharTextGeneratorDataset<DTYPE>::GetVocabSize() {
    return n_Characters;
}
template <typename DTYPE> map<char, int>* CharTextGeneratorDataset<DTYPE>::GetpChar2Index() {
    return m_pChar2Index;
}

#endif
