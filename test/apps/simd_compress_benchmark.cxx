#include "CLI/CLI.hpp"

#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <string>

#include "detdataformats/wib/WIBFrame.hpp"
#include "fdreadoutlibs/wib/tpg/FrameExpand.hpp"
#include "fdreadoutlibs/wib/tpg/TPGConstants.hpp"

using dunedaq::detdataformats::wib::WIBFrame;

const size_t n_channels = 256;
const size_t frame_size = sizeof(WIBFrame);
constexpr size_t REGISTERS_PER_FRAME = swtpg::COLLECTION_REGISTERS_PER_FRAME+swtpg::INDUCTION_REGISTERS_PER_FRAME;

size_t get_file_size(std::string filename)
{
  std::ifstream fin(filename, std::ifstream::binary);
  fin.seekg(0, fin.end);
  size_t length = fin.tellg();
  fin.seekg(0, fin.beg);
  return length;
}


std::vector<char> read_file(std::string filename, size_t max_n_frames=std::numeric_limits<size_t>::max())
{
  std::ifstream fin(filename, std::ifstream::binary);
  size_t length = get_file_size(filename);
  if (length == 0) {
    throw std::runtime_error("Empty file");
  }
  if (length % frame_size != 0) {
    throw std::runtime_error("File does not contain an integer number of frames");
  }
  size_t n_file_frames = length / frame_size;
  size_t n_frames = std::min(n_file_frames, max_n_frames);

  size_t read_length = n_frames * frame_size;
  std::cout << "There are " << n_file_frames << " frames in the file. Running on " << n_frames << std::endl;
  size_t size = n_frames * frame_size;
  size_t size_MB =  size / 1024 / 1024;
  std::cout << "Uncompressed size is " << size_MB << "MB" << std::endl;

  std::vector<char> buffer(read_length);
  fin.read(buffer.data(), read_length);
  return buffer;
  
  // for (size_t i = 0; i < n_frames; ++i) {
  //   fin.seekg(i * frame_size);
  //   // Check we didn't go past the end
  //   if (fin.bad() || fin.eof())
  //     throw std::runtime_error("Error reading file");
  //   // Actually read the fragment into the buffer
  //   fin.read(buffer, frame_size);
  //   WIBFrame* frame = reinterpret_cast<WIBFrame*>(buffer);
  //   for (size_t j = 0; j < n_channels; ++j) {
  //     if (diff && i>0) {
  //       array[i][j] = frame->get_channel(j) - prev_row[j];
  //     } else {
  //       array[i][j] = frame->get_channel(j);
  //     }
  //     prev_row[j] = frame->get_channel(j);
  //   }
  // }
  
}

void print256(__m256i var)
{
    int16_t *val = (int16_t*)&var;
    printf("% 5i % 5i % 5i % 5i % 5i % 5i % 5i % 5i ",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
    val+=8;
    printf("% 5i % 5i % 5i % 5i % 5i % 5i % 5i % 5i",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}

class ExpandedADCView
{
public:
  ExpandedADCView(char* buffer)
    : m_buffer(buffer)
  {}

  __m256i get_register(size_t reg_index, size_t time_index)
  {
    size_t offset = sizeof(__m256i)*(time_index*REGISTERS_PER_FRAME + reg_index);
    return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(m_buffer + offset));
  }
  
private:
  char* m_buffer;
};

__m256i pack4(ExpandedADCView& view, size_t reg_index, size_t time_index)
{
  __m256i reg0 =  view.get_register(reg_index, time_index+0);
  __m256i reg1 =  view.get_register(reg_index, time_index+1);
  __m256i reg2 =  view.get_register(reg_index, time_index+2);
  __m256i reg3 =  view.get_register(reg_index, time_index+3);
  __m256i reg4 =  view.get_register(reg_index, time_index+4);
  // Adjacent-tick differences
  __m256i diff0 = _mm256_sub_epi16(reg1, reg0);
  __m256i diff1 = _mm256_sub_epi16(reg2, reg1);
  __m256i diff2 = _mm256_sub_epi16(reg3, reg2);
  __m256i diff3 = _mm256_sub_epi16(reg4, reg3);

  // Add 8 to everything to ensure all values are positive
  __m256i offset = _mm256_set1_epi16(8u);
  __m256i added0 = _mm256_add_epi16(diff0, offset);
  __m256i added1 = _mm256_add_epi16(diff1, offset);
  __m256i added2 = _mm256_add_epi16(diff2, offset);
  __m256i added3 = _mm256_add_epi16(diff3, offset);
  
  __m256i shifted0 = added0;
  __m256i shifted1 = _mm256_slli_epi16(added1, 4);
  __m256i shifted2 = _mm256_slli_epi16(added2, 8);
  __m256i shifted3 = _mm256_slli_epi16(added3, 12);

  __m256i or_01 = _mm256_or_si256(shifted0, shifted1);
  __m256i or_23 = _mm256_or_si256(shifted2, shifted3);
  return _mm256_or_si256(or_01, or_23);
}

void unpack4(__m256i packed, __m256i* prev,  __m256i* output)
{
  __m256i masked0 = _mm256_and_si256(packed, _mm256_set1_epi16(0xfu));
  __m256i masked1 = _mm256_and_si256(packed, _mm256_set1_epi16(0xf0u));
  __m256i masked2 = _mm256_and_si256(packed, _mm256_set1_epi16(0xf00u));
  __m256i masked3 = _mm256_and_si256(packed, _mm256_set1_epi16(0xf000));

  __m256i shifted0 = masked0;
  __m256i shifted1 = _mm256_srli_epi16(masked1, 4);
  __m256i shifted2 = _mm256_srli_epi16(masked2, 8);
  __m256i shifted3 = _mm256_srli_epi16(masked3, 12);
  
  __m256i offset = _mm256_set1_epi16(8u);

  __m256i subtracted0 = _mm256_sub_epi16(shifted0, offset);
  __m256i subtracted1 = _mm256_sub_epi16(shifted1, offset);
  __m256i subtracted2 = _mm256_sub_epi16(shifted2, offset);
  __m256i subtracted3 = _mm256_sub_epi16(shifted3, offset);

  __m256i adc0 = _mm256_add_epi16(subtracted0, _mm256_lddqu_si256(prev));
  __m256i adc1 = _mm256_add_epi16(subtracted1, adc0);
  __m256i adc2 = _mm256_add_epi16(subtracted2, adc1);
  __m256i adc3 = _mm256_add_epi16(subtracted3, adc2);

  _mm256_storeu_si256(prev, adc3);
  
  _mm256_storeu_si256(output + 0, adc0);
  _mm256_storeu_si256(output + 1, adc1);
  _mm256_storeu_si256(output + 2, adc2);
  _mm256_storeu_si256(output + 3, adc3);
}

__m256i pack3(ExpandedADCView& view, size_t reg_index, size_t time_index)
{
  __m256i reg0 =  view.get_register(reg_index, time_index+0);
  __m256i reg1 =  view.get_register(reg_index, time_index+1);
  __m256i reg2 =  view.get_register(reg_index, time_index+2);
  __m256i reg3 =  view.get_register(reg_index, time_index+3);
  // Adjacent-tick differences
  __m256i diff0 = _mm256_sub_epi16(reg1, reg0);
  __m256i diff1 = _mm256_sub_epi16(reg2, reg1);
  __m256i diff2 = _mm256_sub_epi16(reg3, reg2);

  // Add 8 to everything to ensure all values are positive
  __m256i offset = _mm256_set1_epi16(16);
  __m256i added0 = _mm256_add_epi16(diff0, offset);
  __m256i added1 = _mm256_add_epi16(diff1, offset);
  __m256i added2 = _mm256_add_epi16(diff2, offset);
  
  __m256i shifted0 = added0;
  __m256i shifted1 = _mm256_slli_epi16(added1, 5);
  __m256i shifted2 = _mm256_slli_epi16(added2, 10);

  __m256i or_01 = _mm256_or_si256(shifted0, shifted1);
  return _mm256_or_si256(or_01, shifted2);
}

void unpack3(__m256i packed,  __m256i* prev, __m256i* output)
{
  __m256i masked0 = _mm256_and_si256(packed, _mm256_set1_epi16(0x1f));
  __m256i masked1 = _mm256_and_si256(packed, _mm256_set1_epi16(0x1f << 5));
  __m256i masked2 = _mm256_and_si256(packed, _mm256_set1_epi16(0x1f << 10));

  __m256i shifted0 = masked0;
  __m256i shifted1 = _mm256_srli_epi16(masked1, 5);
  __m256i shifted2 = _mm256_srli_epi16(masked2, 10);
  
  __m256i offset = _mm256_set1_epi16(16);

  __m256i subtracted0 = _mm256_sub_epi16(shifted0, offset);
  __m256i subtracted1 = _mm256_sub_epi16(shifted1, offset);
  __m256i subtracted2 = _mm256_sub_epi16(shifted2, offset);

  __m256i adc0 = _mm256_add_epi16(subtracted0, _mm256_lddqu_si256(prev));
  __m256i adc1 = _mm256_add_epi16(subtracted1, adc0);
  __m256i adc2 = _mm256_add_epi16(subtracted2, adc1);

  _mm256_storeu_si256(prev, adc2);
  
  _mm256_storeu_si256(output + 0, adc0);
  _mm256_storeu_si256(output + 1, adc1);
  _mm256_storeu_si256(output + 2, adc2);
}

__m256i pack2(ExpandedADCView& view, size_t reg_index, size_t time_index)
{
  __m256i reg0 =  view.get_register(reg_index, time_index+0);
  __m256i reg1 =  view.get_register(reg_index, time_index+1);
  __m256i reg2 =  view.get_register(reg_index, time_index+2);
  // Adjacent-tick differences
  __m256i diff0 = _mm256_sub_epi16(reg1, reg0);
  __m256i diff1 = _mm256_sub_epi16(reg2, reg1);

  // Add 8 to everything to ensure all values are positive
  __m256i offset = _mm256_set1_epi16(128);
  __m256i added0 = _mm256_add_epi16(diff0, offset);
  __m256i added1 = _mm256_add_epi16(diff1, offset);
  
  __m256i shifted0 = added0;
  __m256i shifted1 = _mm256_slli_epi16(added1, 8);

  return _mm256_or_si256(shifted0, shifted1);
}

void unpack2(__m256i packed, __m256i* prev, __m256i* output)
{
  __m256i masked0 = _mm256_and_si256(packed, _mm256_set1_epi16(0xff));
  __m256i masked1 = _mm256_and_si256(packed, _mm256_set1_epi16(0xff00));

  __m256i shifted0 = masked0;
  __m256i shifted1 = _mm256_srli_epi16(masked1, 8);
  
  __m256i offset = _mm256_set1_epi16(128);

  __m256i subtracted0 = _mm256_sub_epi16(shifted0, offset);
  __m256i subtracted1 = _mm256_sub_epi16(shifted1, offset);

  __m256i adc0 = _mm256_add_epi16(subtracted0, _mm256_lddqu_si256(prev));
  __m256i adc1 = _mm256_add_epi16(subtracted1, adc0);

  _mm256_storeu_si256(prev, adc1);
  
  _mm256_storeu_si256(output + 0, adc0);
  _mm256_storeu_si256(output + 1, adc1);
}

// Starting at register `reg_index`, time `time_index` in the `view` object, how many of the next registers can we pack into one?
size_t get_n_registers(ExpandedADCView& view, size_t reg_index, size_t time_index)
{
  // The next five registers in time
  __m256i reg0 =  view.get_register(reg_index, time_index+0);
  __m256i reg1 =  view.get_register(reg_index, time_index+1);
  __m256i reg2 =  view.get_register(reg_index, time_index+2);
  __m256i reg3 =  view.get_register(reg_index, time_index+3);
  __m256i reg4 =  view.get_register(reg_index, time_index+4);
  // Adjacent-tick differences
  __m256i diff0 = _mm256_sub_epi16(reg1, reg0);
  __m256i diff1 = _mm256_sub_epi16(reg2, reg1);
  __m256i diff2 = _mm256_sub_epi16(reg3, reg2);
  __m256i diff3 = _mm256_sub_epi16(reg4, reg3);

  // Absolute values of differences
  __m256i absdiff0 = _mm256_abs_epi16(diff0);
  __m256i absdiff1 = _mm256_abs_epi16(diff1);
  __m256i absdiff2 = _mm256_abs_epi16(diff2);
  __m256i absdiff3 = _mm256_abs_epi16(diff3);

  // ---------------------------------------------------------
  // Can we fit 4 registers into 1?
  __m256i or_01 = _mm256_or_si256(absdiff0, absdiff1);
  __m256i or_23 = _mm256_or_si256(absdiff2, absdiff3);
  __m256i or_0123 = _mm256_or_si256(or_01, or_23);

  // This register will be all zeros if we _can_ fit it
  __m256i shifted = _mm256_srli_epi16(or_0123, 3);
  int fits = _mm256_testc_si256(_mm256_setzero_si256(), shifted);
  if (fits) {
    return 4;
  }

  // ---------------------------------------------------------
  // Can we fit 3 registers into 1?
  __m256i or_012 = _mm256_or_si256(or_01, absdiff2);
  shifted = _mm256_srli_epi16(or_012, 4);
  fits = _mm256_testc_si256(_mm256_setzero_si256(), shifted);
  if (fits) {
    return 3;
  }

  // ---------------------------------------------------------
  // Can we fit 2 registers into 1?
  shifted = _mm256_srli_epi16(or_01, 7);
  fits = _mm256_testc_si256(_mm256_setzero_si256(), shifted);
  if (fits) {
    return 2;
  }

  return 1;
}

int
main(int argc, char** argv)
{
  CLI::App app{ "Write raw WIB data as HDF5 file in fragment style" };

  std::string in_filename;
  app.add_option("-i,--input", in_filename, "Input raw WIB file");

  size_t max_n_frames=std::numeric_limits<size_t>::max();
  app.add_option("-n,--max-n-frames", max_n_frames, "Maximum number of frames to write");

  CLI11_PARSE(app, argc, argv);

  const size_t digitization_freq = 2000000;
  std::vector<char> uncompressed = read_file(in_filename, max_n_frames);
  size_t n_frames = uncompressed.size()/frame_size;
  double data_duration_ms = double(n_frames)/digitization_freq*1000;
  size_t expanded_size = (256/8)*REGISTERS_PER_FRAME*n_frames;
  std::vector<char> expanded(expanded_size);
  WIBFrame* frame = reinterpret_cast<WIBFrame*>(uncompressed.data());
  std::cout << frame->get_timestamp() << std::endl;
  size_t expanded_frame_size = 32*REGISTERS_PER_FRAME;

  using namespace std::chrono;

  auto start = steady_clock::now();
  for (size_t i=0; i<n_frames; ++i) {
    swtpg::RegisterArray<REGISTERS_PER_FRAME> expanded_frame = swtpg::get_frame_all_adcs(frame);
    memcpy(expanded.data()+i*expanded_frame_size,
           expanded_frame.data(),
           expanded_frame_size);
    ++frame;
  }
  auto end = steady_clock::now();
  auto dur_ms = duration_cast<milliseconds>(end-start).count();
  std::cout << "Expanded and copied " << data_duration_ms << "ms of data in " << dur_ms << "ms" << std::endl;

  // uint16_t* adc = reinterpret_cast<uint16_t*>(expanded.data());
  // for (int i=0; i<1000; ++i) {
  //   std::cout << (*adc) << " ";
  //   ++adc;
  // }
  // std::cout << std::endl;

  ExpandedADCView view(expanded.data());
  // __m256i prev = view.get_register(0, 0);
  // for (int i=1; i<24; ++i) {
  //   __m256i cur = view.get_register(0, i);
  //   __m256i diff = _mm256_sub_epi16(cur, prev);
  //   print256(diff); printf("\n");
  //   prev = cur;
  //   if (i%4 == 0) {
  //     printf("\n");
  //   }
  // }
  // for (int i=0; i<6; ++i) {
  //   std::cout << get_n_registers(view, 0, 4*i) << std::endl;
  // }

  // --------------------------------------------------------
  // Pack one register for testing

  int ns[n_frames]={0};
  __m256i* packed = new __m256i[n_frames*REGISTERS_PER_FRAME];
  for (size_t i=0; i<n_frames*REGISTERS_PER_FRAME; ++i) _mm256_storeu_si256(packed+i, _mm256_setzero_si256());

  auto start_pack = steady_clock::now();
  packed[0] = view.get_register(0, 0);
  ns[0] = 16; // Magic value for "first time sample in this register"
  size_t t = 0;
  size_t packed_index = 1;
  while (t < n_frames - 4) {
    int n = get_n_registers(view, 0, t);
    // std::cout << "Packing " << n << " registers starting at time " << t << std::endl;
    ns[packed_index] = n;
    switch (n) {
    case 4:
      packed[packed_index] = pack4(view, 0, t);
      break;
    case 3:
      packed[packed_index] = pack3(view, 0, t);
      break;
    case 2:
      packed[packed_index] = pack2(view, 0, t);
      break;
    case 1:
      packed[packed_index] = _mm256_sub_epi16(view.get_register(0, t+1), view.get_register(0, t));
      break;
    }
    // print256(packed[packed_index]); printf("\n");
    t+=n;
    ++packed_index;
  }
  // Put the last few samples in the output unchanged
  while (t < n_frames) {
    ns[packed_index] = 1;
    packed[packed_index] = _mm256_sub_epi16(view.get_register(0, t+1), view.get_register(0, t));
    t += 1;
    ++packed_index;
  }
  ns[packed_index] = 0;
  auto end_pack = steady_clock::now();
  auto dur_pack_ms = duration_cast<milliseconds>(end_pack-start_pack).count();
  std::cout << "Packed " << n_frames << " frames into " << packed_index << " in " << dur_pack_ms << "ms" << std::endl;

  // --------------------------------------------------------
  // Unpack the packed data

  __m256i* unpacked = new __m256i[n_frames*REGISTERS_PER_FRAME];
  for (int i=0; i<n_frames*REGISTERS_PER_FRAME; ++i) _mm256_storeu_si256(unpacked+i, _mm256_setzero_si256());
  
  // First value is the un-diffed register values
  __m256i prev = packed[0];
  _mm256_storeu_si256(unpacked, packed[0]);
  size_t i = 1;
  size_t output_time_sample = 1;
  while (ns[i] != 0 ) {
    switch (ns[i]) {
    case 4:
      unpack4(packed[i], &prev, unpacked+output_time_sample);
      break;
    case 3:
      unpack3(packed[i], &prev, unpacked+output_time_sample);
      break;
    case 2:
      unpack2(packed[i], &prev, unpacked+output_time_sample);
      break;
    case 1:
      // printf("Unpack1\n");
      // printf("i=%zu packed[i]: ", i); print256(packed[i]); printf("\n");
      // printf("prev before: "); print256(prev); printf("\n");
      prev = _mm256_add_epi16(prev, packed[i]);
      // printf("prev after:  "); print256(prev); printf("\n");
      _mm256_storeu_si256(unpacked+output_time_sample, prev);
      break;
    }
    output_time_sample += ns[i];
    ++i;
  }

  printf("Original:\n");
  for (int i=0; i<20; ++i) {
    printf("t=% 3d: ", i); print256(view.get_register(0, i)); printf("\n");
    if (i%4 == 0) {
      printf("\n");
    }
  }
  
  printf("Unpacked:\n");
  for (int i=0; i<20; ++i) {
    printf("t=% 3d: ", i); print256(unpacked[i]); printf("\n");
    if (i%4 == 0) {
      printf("\n");
    }
  }

  bool passed = true;
  for (size_t i=0; i<n_frames; ++i) {
    __m256i orig = view.get_register(0, i);
    __m256i roundtrip = unpacked[i];
    __m256i diff = _mm256_sub_epi16(orig, roundtrip);
    int all_zero = _mm256_testc_si256(_mm256_setzero_si256(), diff);
    if (!all_zero) {
      std::cout << "At sample " << i << " orig/roundtrip:" << std::endl;
      print256(orig); printf("\n");
      print256(roundtrip); printf("\n");
      passed = false;
      break;
    }
  }
  if (passed) {
    std::cout << "All samples were identical after roundtrip" << std::endl;
  }
  
  delete[] packed;
  delete[] unpacked;
  return 0;
}
