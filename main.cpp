#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <boost/mpi.hpp>

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/signal_handler.h"

using std::pair;
using std::vector;
using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::SGDSolver;
using caffe::shared_ptr;
namespace mpi = boost::mpi;

pair<vector<float>, int> load_image(const string& path);
pair<vector<float>, int> load_label(const string& path);

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  mpi::environment env;
  mpi::communicator world;

  Caffe::set_mode(Caffe::CPU);
  caffe::SignalHandler signal_handler(caffe::SolverAction::STOP, caffe::SolverAction::SNAPSHOT);
  
  auto images = load_image("model/train-images-idx3-ubyte");
  auto labels = load_label("model/train-labels-idx1-ubyte");
  auto testim = load_image("model/t10k-images-idx3-ubyte");
  auto testlb = load_label("model/t10k-labels-idx1-ubyte");
  assert(images.second == labels.second);
  
  caffe::SolverParameter solver_param1;
  caffe::ReadSolverParamsFromTextFileOrDie("model/mnist_solver.prototxt", &solver_param1);
  shared_ptr<Solver<float> > 
       solver1(caffe::SolverRegistry<float>::CreateSolver(solver_param1));
  solver1->SetActionFunction(signal_handler.GetActionFunction());
   
  auto solverNet = solver1->net();
  auto solver_memory_data = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(solverNet->layer_by_name("mnist"));
  solver_memory_data->Reset(images.first.data(), labels.first.data(), 60000/64*64);
  // auto tester_memory_data = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(testerNet->layer_by_name("mnist"));
  // tester_memory_data->Reset(testim.first.data(), testlb.first.data(), 10000/64*64);
  if (world.rank() == 0)
    std::cout << "Me" << std::endl;
  return 0;
}

unsigned int be(const char *buf) {
  char buf2[4];
  buf2[0] = buf[3];
  buf2[1] = buf[2];
  buf2[2] = buf[1];
  buf2[3] = buf[0];
  return *(unsigned int *)buf2;
}

using std::ifstream;
using std::istreambuf_iterator;
using std::transform;

pair<vector<float>, int> load_image(const string& path) {
  char buf[1000];
  ifstream input(path, std::ios::binary);
  input.read(buf, 4);
  assert(be(buf) == 0x00000803);
  input.read(buf, 12);
  int number = be(buf);
  int rows   = be(buf+4);
  int cols   = be(buf+8);
  int total  = number * rows * cols;
  vector<float> alloc(total);
  transform(
    (istreambuf_iterator<char>(input)),
    istreambuf_iterator<char>(),
    alloc.begin(),
    [](unsigned char c) -> float {return ((float) c) / 255.f;});
  return std::make_pair(alloc, number);
}

pair<vector<float>, int> load_label(const string& path) {
  char buf[10];
  ifstream input(path, std::ios::binary);
  input.read(buf, 4);
  assert(be(buf) == 0x00000801);
  input.read(buf, 4);
  int total = be(buf);
  vector<float> alloc(total);
  transform(
    (istreambuf_iterator<char>(input)),
    istreambuf_iterator<char>(),
    alloc.begin(),
    [](unsigned char c) -> float {return (float) c; });
  return std::make_pair(alloc, total);
}
