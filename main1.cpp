#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "FEM1.h"
#include "writeSolutions.h"

using namespace dealii;

int main() {
    try {
        deallog.depth_console(0); // ��������� ������� (������ ����)

        for (unsigned int order = 1; order <= 3; order++) {  //Specify the basis function order: 1, 2, or 3
        // order - ������� �������� ������� �������� == ����� ����� � �������� - 1(������������ � ������������� ������� � ������� ��������)
            for (unsigned int problem = 1; problem <= 2; problem++) {  //Specify the subproblem: 1 or 2
                // problem - ����� ������ (1 - ������ �������-�������, � ����� 0 � L ������ ������� �������; 2 - ������ �������-��� �������)
                FEM<1> problemObject(order, problem); // �������� ������� �������� (��� ��� ���� ����������� ��� ������); 1 - ����������� ������ (� ������ ���� ������ 1)
                // std::cout << "==============Object created============" << std::endl; // !!!!!

                //Define the number of elements as an input to "generate_mesh"
                problemObject.generate_mesh(10); //e.g. a 10 element mesh (�������� �����, ��������� ������� ����� ������� �� 10 ������ �� ����� �������� ���������)
                // std::cout << "==============generate_mesh=============" << std::endl; // !!!!!
                problemObject.setup_system(); // ��������� �������� ������, ��������, ����������� ���������� ������������ ������ ��������� ������� �������� ��� ���������� ������������ ��������� 
                // std::cout << "==============setup_system==============" << std::endl; // !!!!!
                problemObject.assemble_system(); // ��������������� (������� �� ������������ �� �������� ��������� � ��������� ������ (��������� ������))
                // std::cout << "==============assemble_system===========" << std::endl; // !!!!!

                // std::cout << "prob: " << problemObject.prob << std::endl; // !!!!!
                // std::cout << "basisFunctionOrder: " << problemObject.basisFunctionOrder << std::endl; // !!!!!
                // std::cout << "L: " << problemObject.L << std::endl; // !!!!!
                // std::cout << "g1: " << problemObject.g1 << std::endl; // !!!!!
                // std::cout << "g2: " << problemObject.g2 << std::endl; // !!!!!
                // std::cout << "SparseMatrix K n_rows: " << problemObject.K.get_sparsity_pattern().n_rows() << std::endl; // !!!!!
                // std::cout << "SparseMatrix K n_cols: " << problemObject.K.get_sparsity_pattern().n_cols() << std::endl; // !!!!!

                // ����� ������� K
                // problemObject.K.print(std::cout, false, false);

                // ����� sparsity_pattern
                // for (unsigned int i = 0; i < problemObject.K.get_sparsity_pattern().n_rows(); i++) { // !!!!!
                //   for (unsigned int j = 0; j < problemObject.K.get_sparsity_pattern().n_cols(); j++){ // !!!!!
                //     std::cout << problemObject.K.get_sparsity_pattern()(i,j) << "\t"; // !!!!!
                //   } // !!!!!
                //   std::cout << std::endl; // !!!!!
                // } // !!!!!

                // ����� ������� F
                // std::cout << "������ F:" << std::endl;
                // for (unsigned int i = 0; i < problemObject.K.get_sparsity_pattern().n_rows(); i++)
                //   std::cout << problemObject.F[i] << "\t";
                // std::cout << std::endl;

                problemObject.solve(); // ������� ������� �������� ���������, � ������� �������� ������
                // std::cout << "=================solve==================" << std::endl; // !!!!!

                // ����� ������� D
                // std::cout << "������ D:" << std::endl;
                // for (unsigned int i = 0; i < problemObject.K.get_sparsity_pattern().n_cols(); i++)
                //   std::cout << problemObject.D[i] << "\t";
                // std::cout << std::endl;

                std::cout << "Order: " << int(order) << ", problem: " << int(problem) << ", l2 norm of error: " << problemObject.l2norm_of_error() << std::endl; // ����� ����� ������ (���� ����, �� ������ ���������� �������-���������� ������� ���������� �� ������� ��������������)

                //write output file in vtk format for visualization
                problemObject.output_results(); // ����� ����������� (������ �� ���� ������?)

                //write solutions to h5 file (vtk �����, ��� ����� ����������� � ������� ��������� paraview)
                char tag[21];
                sprintf(tag, "CA1_Order%d_Problem%d", order, problem);
                writeSolutionsToFileCA1(problemObject.D, problemObject.l2norm_of_error(), tag);
            }
        }
    }
    catch (std::exception& exc) {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    return 0;
}