package parakeet

import (
	"errors"
	"math"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestAudioValidation(t *testing.T) {
	for _, test := range []struct {
		request Request
		want    error
	}{
		{Request{make([]float32, 320), 16000}, nil},
		{Request{nil, 16000}, ErrInvalidAudio},
		{Request{make([]float32, 319), 16000}, ErrInvalidAudio},
		{Request{make([]float32, 320), 24000}, ErrInvalidAudio},
		{Request{make([]float32, MaxSamples+1), 16000}, ErrAudioTooLong},
	} {
		if err := validateAudio(test.request); !errors.Is(err, test.want) {
			t.Fatalf("%v, wanted %v", err, test.want)
		}
	}
	samples := make([]float32, 320)
	samples[319] = float32(math.NaN())
	if err := validateAudio(Request{samples, 16000}); !errors.Is(err, ErrInvalidAudio) {
		t.Fatal(err)
	}
}

func TestMetadataValidation(t *testing.T) {
	info := []ort.InputOutputInfo{
		{Name: "waveforms", OrtValueType: ort.ONNXTypeTensor, DataType: ort.TensorElementDataTypeFloat, Dimensions: ort.NewShape(-1, -1)},
		{Name: "waveforms_lens", OrtValueType: ort.ONNXTypeTensor, DataType: ort.TensorElementDataTypeInt64, Dimensions: ort.NewShape(-1)},
	}
	if _, err := validateTensors(info, preInputs); err != nil {
		t.Fatal(err)
	}
	info[1].DataType = ort.TensorElementDataTypeInt32
	if _, err := validateTensors(info, preInputs); err == nil {
		t.Fatal("accepted wrong length type")
	}
	info[1].DataType = ort.TensorElementDataTypeInt64
	info[1].Name = "waveforms"
	if _, err := validateTensors(info, preInputs); err == nil {
		t.Fatal("accepted duplicate names")
	}
}
