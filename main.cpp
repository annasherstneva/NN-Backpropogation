#include <cmath>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "NeuralNetw.h"
#include <fstream>
#include <string>
 
using namespace std;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(vector<vector<double>> &arr, std::string name)
{
	ifstream file(name, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		arr.resize(number_of_images);

		for (int i = 0;i<number_of_images;++i)
		{
			for (int r = 0;r<n_rows;++r)
			{
				for (int c = 0;c<n_cols;++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					arr[i].push_back((double)temp/255.0);
				}
			}
		}
	}
}

void ReadLabelsMNIST(vector<double> &arr, std::string name)
{
	ifstream file(name, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0;i<number_of_images;++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			arr.push_back((double)temp);
		}
	}
}

int main(int argc, char** argv)
{
	int input;
	int output;
	int hide;
	double speed = 0.08;
	if (argc < 5) {
		cout << "Input necessary arguments" << endl;
		cout << "Path to every Mnist sets / labels with it names if form like this: 'C:\\t10k-images.idx3-ubyte' " << endl;
		cout << "Use next order: " << endl;
		cout << "Path to Mnist train images " << endl;
		cout << "Path to Mnist train labels " << endl;
		cout << "Path to Mnist test images " << endl;
		cout << "Path to Mnist test labels " << endl;
		cout << "Speed of learning (by default speed = 0.08)" << endl;
		return 0;
	}
	

	vector <vector <double>> Dataset;
	vector <double> Labels;

	vector <vector <double>> Testset;
	vector <double> TestLabels;
	string test, train, testl, trainl;

	train = argv[1];
	trainl = argv[2];
	test = argv[3];
	testl = argv[4];

	if (argc > 5)
		speed = atof(argv[5]);

	ReadMNIST(Testset, test);
	ReadLabelsMNIST(TestLabels, testl);

	ReadMNIST(Dataset, train);
	ReadLabelsMNIST(Labels, trainl);

	input = 28 * 28;
	hide = 350;
	output = 10;


	cout << "Creating NN..." << endl;
	NeuralNetw NN = NeuralNetw(input, output, hide,speed);

	cout << "Train set:" << endl;
	NN.Train(Dataset, Labels);
	cout << endl;
	cout << "Test set:" << endl;
	NN.Train(Testset, TestLabels);
	system("pause");
	return 0;
}

 
 