package parakeet

import (
	"errors"
	"fmt"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

func sessionOptions(c Config, provider Provider) (_ *ort.SessionOptions, err error) {
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, o.Destroy())
		}
	}()
	threads := c.Threads
	if threads == 0 {
		threads = 1
	}
	for _, set := range []func() error{
		func() error { return o.SetIntraOpNumThreads(threads) },
		func() error { return o.SetInterOpNumThreads(1) },
		func() error { return o.SetExecutionMode(ort.ExecutionModeSequential) },
		func() error { return o.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll) },
	} {
		if err = set(); err != nil {
			return nil, err
		}
	}
	if c.Verbose {
		if err = o.SetLogSeverityLevel(ort.LoggingLevelVerbose); err != nil {
			return nil, err
		}
	}
	if provider == CUDA {
		cuda, e := ort.NewCUDAProviderOptions()
		if e != nil {
			return nil, fmt.Errorf("create CUDA options: %w", e)
		}
		err = cuda.Update(map[string]string{
			"device_id": strconv.Itoa(c.DeviceID), "cudnn_conv_algo_search": "DEFAULT",
			"use_tf32": "0", "arena_extend_strategy": "kSameAsRequested",
		})
		if err == nil {
			err = o.AppendExecutionProviderCUDA(cuda)
		}
		err = errors.Join(err, cuda.Destroy())
		if err != nil {
			return nil, fmt.Errorf("configure CUDA provider: %w", err)
		}
	}
	return o, nil
}

type tensorSpec struct {
	name  string
	kind  ort.TensorElementDataType
	shape []int64
}

var (
	preInputs = []tensorSpec{
		{"waveforms", ort.TensorElementDataTypeFloat, []int64{-1, -1}},
		{"waveforms_lens", ort.TensorElementDataTypeInt64, []int64{-1}},
	}
	preOutputs = []tensorSpec{
		{"features", ort.TensorElementDataTypeFloat, []int64{-1, 128, -1}},
		{"features_lens", ort.TensorElementDataTypeInt64, []int64{-1}},
	}
	encInputs = []tensorSpec{
		{"audio_signal", ort.TensorElementDataTypeFloat, []int64{-1, 128, -1}},
		{"length", ort.TensorElementDataTypeInt64, []int64{-1}},
	}
	encOutputs = []tensorSpec{
		{"outputs", ort.TensorElementDataTypeFloat, []int64{-1, 1024, -1}},
		{"encoded_lengths", ort.TensorElementDataTypeInt64, []int64{-1}},
	}
	decInputs = []tensorSpec{
		{"encoder_outputs", ort.TensorElementDataTypeFloat, []int64{-1, 1024, -1}},
		{"targets", ort.TensorElementDataTypeInt32, []int64{-1, -1}},
		{"target_length", ort.TensorElementDataTypeInt32, []int64{-1}},
		{"input_states_1", ort.TensorElementDataTypeFloat, []int64{2, -1, 640}},
		{"input_states_2", ort.TensorElementDataTypeFloat, []int64{2, -1, 640}},
	}
	decOutputs = []tensorSpec{
		{"outputs", ort.TensorElementDataTypeFloat, []int64{-1, -1, -1, 8198}},
		{"prednet_lengths", ort.TensorElementDataTypeInt32, []int64{-1}},
		{"output_states_1", ort.TensorElementDataTypeFloat, []int64{2, -1, 640}},
		{"output_states_2", ort.TensorElementDataTypeFloat, []int64{2, -1, 640}},
	}
)

func names(specs []tensorSpec) []string {
	result := make([]string, len(specs))
	for i, spec := range specs {
		result[i] = spec.name
	}
	return result
}

func validateTensors(info []ort.InputOutputInfo, specs []tensorSpec) ([]TensorInfo, error) {
	if len(info) != len(specs) {
		return nil, fmt.Errorf("expected %d tensors, got %d", len(specs), len(info))
	}
	seen := make(map[string]bool)
	result := make([]TensorInfo, 0, len(info))
	for _, tensor := range info {
		var spec *tensorSpec
		for i := range specs {
			if specs[i].name == tensor.Name {
				spec = &specs[i]
				break
			}
		}
		if spec == nil || seen[tensor.Name] || tensor.OrtValueType != ort.ONNXTypeTensor || tensor.DataType != spec.kind || len(tensor.Dimensions) != len(spec.shape) {
			return nil, fmt.Errorf("incompatible tensor %q", tensor.Name)
		}
		for i, dim := range spec.shape {
			actual := tensor.Dimensions[i]
			if (dim > 0 && actual != dim) || (dim < 0 && actual >= 0) {
				return nil, fmt.Errorf("incompatible shape for %q: %v", tensor.Name, tensor.Dimensions)
			}
		}
		seen[tensor.Name] = true
		result = append(result, TensorInfo{tensor.Name, append([]int64(nil), tensor.Dimensions...), tensor.DataType.String()})
	}
	return result, nil
}

func inspect(path string, options *ort.SessionOptions, inputs, outputs []tensorSpec) (GraphInfo, error) {
	in, out, err := ort.GetInputOutputInfoWithOptions(path, options)
	if err != nil {
		return GraphInfo{}, fmt.Errorf("inspect %s: %w", path, err)
	}
	info := GraphInfo{}
	info.Inputs, err = validateTensors(in, inputs)
	if err == nil {
		info.Outputs, err = validateTensors(out, outputs)
	}
	if err != nil {
		return GraphInfo{}, fmt.Errorf("validate %s: %w", path, err)
	}
	return info, nil
}

func loadDynamic(path string, options *ort.SessionOptions, in, out []tensorSpec) (*ort.DynamicAdvancedSession, GraphInfo, error) {
	info, err := inspect(path, options, in, out)
	if err != nil {
		return nil, GraphInfo{}, err
	}
	session, err := ort.NewDynamicAdvancedSession(path, names(in), names(out), options)
	if err != nil {
		return nil, GraphInfo{}, fmt.Errorf("load %s: %w", path, err)
	}
	return session, info, nil
}
