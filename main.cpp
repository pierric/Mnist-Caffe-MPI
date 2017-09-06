#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <random>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <ctime>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>

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

pair<vector<float>, pair<int,int>> load_image(const string& path);
pair<vector<float>, int> load_label(const string& path);

class BlobCollection {
public:
  BlobCollection();
  void load(const vector<caffe::shared_ptr<Blob<float> > >& params);
  void save(const vector<caffe::shared_ptr<Blob<float> > >& params);
  void add_or_copy(const BlobCollection& other);
  void divide_by(int n);
  int size() const { return _blobs.size(); }

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive & ar, const unsigned int version) {
    ar & _blobs;
  }
private:
  vector<vector<float> > _blobs;
};
BOOST_CLASS_VERSION(BlobCollection, 1)

BlobCollection::BlobCollection() {
}

void BlobCollection::add_or_copy(const BlobCollection& other) {
  if (_blobs.empty()) {
    _blobs = other._blobs;
    return;
  }
  if (_blobs.size() != other._blobs.size())
    throw std::logic_error("Cannot add by different number of Blobs");
  for (int i=0; i<_blobs.size(); ++i) {
    auto &src = _blobs[i];
    auto &dst = other._blobs[i];
    if (src.size() != dst.size()) 
      throw std::logic_error("Cannot add two blobs of different size");
    for (int j=0; j<src.size(); ++j) {
      src[j] += dst[j];
    }
  }
}

void BlobCollection::divide_by(int n) {
  for (auto blob=_blobs.begin(); blob!=_blobs.end(); ++blob) {
    for (int i=0; i < blob->size(); ++i) {
      (*blob)[i] /= n;
    }
  }
}

void BlobCollection::load(const vector<caffe::shared_ptr<Blob<float> > >& params) {
  if (_blobs.empty()) {
    for (auto it=params.begin(); it != params.end(); ++it) {
      const float *databeg = (*it)->cpu_data();
      const float *dataend = databeg + (*it)->count();
      _blobs.push_back(vector<float>(databeg, dataend));
    }
  }
  else 
    throw std::logic_error("BlobCollection is alread loaded.");
}

void BlobCollection::save(const vector<caffe::shared_ptr<Blob<float> > >& params) {
  if (_blobs.size() != params.size())
    throw std::logic_error("BlobCollection has different number of blobs.");
  for (int i=0;i<_blobs.size();++i) {
    float *data = params[i]->mutable_cpu_data();
    if (_blobs[i].size() != params[i]->count()) {
      throw std::logic_error("Blob cannot be saved, for it has different size.");
    }
    copy(_blobs[i].begin(), _blobs[i].end(), data);
  }
}

const int NUM_ITERATIONS = 10;
const int NUM_BATCHES_PER_ITER = 500;
const int NUM_ITERS_PER_TEST = 3;

std::ostream& out(mpi::communicator &world) {
  using namespace std;
  time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
  char buf[64];
  strftime(buf, 64, "%T", localtime(&now));
  return cout << "[" << world.rank() << "] " 
              << fixed << setprecision(2)
              << buf
              << " ";
}

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
  assert(images.second.first == labels.second);
  
  using std::endl;
  out(world) << "mnist loaded" << endl;

  // create a solver for learning
  //   shuffle the dataset, and fill in the input of this solver
  caffe::SolverParameter solver_param1;
  caffe::ReadSolverParamsFromTextFileOrDie("model/mnist_solver.prototxt", &solver_param1);
  shared_ptr<Solver<float> > 
       solver1(caffe::SolverRegistry<float>::CreateSolver(solver_param1));
  solver1->SetActionFunction(signal_handler.GetActionFunction());
   
  auto solverNet = solver1->net();
  auto solver_memory_data = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(solverNet->layer_by_name("mnist"));
  int batch_size = solver_memory_data->batch_size();

  int img_number = images.second.first;
  int img_size   = images.second.second;
  std::random_device rd;
  std::mt19937 gen(rd());
  typedef std::uniform_int_distribution<int> distr_t;
  typedef typename distr_t::param_type param_t;
  distr_t D;
  for (int i=img_number-1; i>0; --i) {
    int j = D(gen, param_t(0, i));
    auto src = images.first.begin() + i*img_size;
    auto dst = images.first.begin() + j*img_size;
    std::swap_ranges(src, src+img_size, dst);
    std::swap(labels.first[i], labels.first[j]);
  }
  solver_memory_data->Reset(images.first.data(), labels.first.data(), img_number/batch_size*batch_size);

  // create a network for testing
  //   it shares the parameters with the previous solver.
  //   we don't pre-fill it in. that will be before each test.
  caffe::NetParameter net_param;
  ReadNetParamsFromTextFileOrDie("model/mnist_net.prototxt", &net_param);
  caffe::Net<float> testerNet(net_param);
  auto tester_memory_data = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(testerNet.layer_by_name("mnist"));
  assert(tester_memory_data->batch_size() == batch_size);
  testerNet.ShareTrainedLayersWith(solverNet.get());
  
  out(world) << "data populated" << endl;
  
  for (int iter=0; iter < NUM_ITERATIONS; ++iter) {
    if (world.rank() == 0) {
      out(world) << "iteration " << iter << endl;
    }

    BlobCollection reducedWeights;
    // reducedWeights stores the weights for this training iteration,
    //   in the very first iteration, it is just the solverNet's weights on node 0
    //   afterwards, it is average of the weights of the solverNets from all nodes
    if (iter == 0) {
      if (world.rank() == 0) {
        reducedWeights.load(solverNet->params());
      }
    }
    else {
      vector<BlobCollection> reducedWeightsN;

      BlobCollection weight;
      weight.load(solverNet->params());
        
      gather(world, weight, reducedWeightsN, 0);
      
      if (world.rank() == 0) {
        out(world) << "Weights gathered" << endl;
        for (auto it=reducedWeightsN.begin(); it!=reducedWeightsN.end(); ++it) {
          reducedWeights.add_or_copy(*it);
        }
        reducedWeights.divide_by(reducedWeightsN.size());
      }
    }

    // distribute the reducedWeights to all nodes
    broadcast(world, reducedWeights, 0);
    reducedWeights.save(solverNet->params());

    // do one step of learn
    //   periodically, the master does a accuracy testing on the test dataset.
    if (world.rank() == 0) {
      if (iter % NUM_ITERS_PER_TEST == 0) {
        float accuracy = 0;
        int img_number = testim.second.first;
        int N = img_number / batch_size;
        tester_memory_data->Reset(testim.first.data(), testlb.first.data(), N*batch_size);      
        for (int x = 0; x < N; x++) {
          testerNet.Forward();
          auto blob = testerNet.blob_by_name("accuracy");
          accuracy += blob->data_at(0,0,0,0);
        }
        accuracy /= N;
        out(world) << "==> Master Test accuracy: " << std::setprecision(4) << accuracy << endl;
      }
    }
    solver1->Step(NUM_BATCHES_PER_ITER);
    // test on each slave and each step. for debug purpose.
    // {
    //   float accuracy = 0;
    //   int N = testim.second / batch_size;
    //   tester_memory_data->Reset(testim.first.data(), testlb.first.data(), testim.second/batch_size*batch_size);      
    //   for (int x = 0; x < N; x++) {
    //     testerNet.Forward();
    //     auto blob = testerNet.blob_by_name("accuracy");
    //     accuracy += blob->data_at(0,0,0,0);
    //   }
    //   accuracy /= N;
    //   out(world) << "==> Slave Test accuracy: " << accuracy << endl;
    // }
  }

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

pair<vector<float>, pair<int,int>> load_image(const string& path) {
  char buf[1000];
  ifstream input(path, std::ios::binary);
  input.read(buf, 4);
  assert(be(buf) == 0x00000803);
  input.read(buf, 12);
  int number = be(buf);
  int rows   = be(buf+4);
  int cols   = be(buf+8);
  int size   = rows * cols;
  vector<float> alloc(number * size);
  transform(
    (istreambuf_iterator<char>(input)),
    istreambuf_iterator<char>(),
    alloc.begin(),
    [](unsigned char c) -> float {return ((float) c) / 255.f;});
  return std::make_pair(alloc, std::make_pair(number, size));
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
