package parakeet

import (
	"errors"
	"fmt"

	"github.com/androiddrew/go-parakeet/internal/assets"
	ort "github.com/yalue/onnxruntime_go"
)

type nativeDecoder struct {
	session                           *ort.AdvancedSession
	values                            []ort.Value
	frame, s1, s2, logits, out1, out2 *ort.Tensor[float32]
	target                            *ort.Tensor[int32]
}

func newNativeDecoder(path string, options *ort.SessionOptions) (_ *nativeDecoder, err error) {
	d := &nativeDecoder{}
	defer func() {
		if err != nil {
			err = errors.Join(err, d.close())
		}
	}()
	for _, entry := range []struct {
		target **ort.Tensor[float32]
		shape  ort.Shape
	}{
		{&d.frame, ort.NewShape(1, assets.EncoderDim, 1)},
		{&d.s1, ort.NewShape(assets.StateLayers, 1, assets.StateDim)},
		{&d.s2, ort.NewShape(assets.StateLayers, 1, assets.StateDim)},
		{&d.logits, ort.NewShape(1, 1, 1, assets.VocabSize+assets.Durations)},
		{&d.out1, ort.NewShape(assets.StateLayers, 1, assets.StateDim)},
		{&d.out2, ort.NewShape(assets.StateLayers, 1, assets.StateDim)},
	} {
		*entry.target, err = ort.NewEmptyTensor[float32](entry.shape)
		if err != nil {
			return nil, err
		}
		d.values = append(d.values, *entry.target)
	}
	d.target, err = ort.NewEmptyTensor[int32](ort.NewShape(1, 1))
	if err != nil {
		return nil, err
	}
	d.values = append(d.values, d.target)
	length, err := ort.NewTensor(ort.NewShape(1), []int32{1})
	if err != nil {
		return nil, err
	}
	d.values = append(d.values, length)
	d.session, err = ort.NewAdvancedSession(path, names(decInputs), []string{"outputs", "output_states_1", "output_states_2"},
		[]ort.Value{d.frame, d.target, length, d.s1, d.s2}, []ort.Value{d.logits, d.out1, d.out2}, options)
	if err != nil {
		return nil, fmt.Errorf("load decoder: %w", err)
	}
	return d, nil
}

func (d *nativeDecoder) close() error {
	var err error
	if d.session != nil {
		err = d.session.Destroy()
		d.session = nil
	}
	for _, value := range d.values {
		err = errors.Join(err, value.Destroy())
	}
	d.values = nil
	return err
}

func (d *nativeDecoder) reset() {
	clear(d.s1.GetData())
	clear(d.s2.GetData())
}

func (d *nativeDecoder) step(encoded sequence, frame, previous int) ([]float32, error) {
	for channel := range assets.EncoderDim {
		// Physical stride can exceed the valid length because preprocessing pads.
		d.frame.GetData()[channel] = encoded.data[channel*encoded.stride+frame]
	}
	d.target.GetData()[0] = int32(previous)
	if err := d.session.Run(); err != nil {
		return nil, err
	}
	if err := validateFinite(d.out1.GetData()); err != nil {
		return nil, fmt.Errorf("decoder hidden state: %w", err)
	}
	if err := validateFinite(d.out2.GetData()); err != nil {
		return nil, fmt.Errorf("decoder cell state: %w", err)
	}
	return d.logits.GetData(), nil
}

func (d *nativeDecoder) commit() {
	copy(d.s1.GetData(), d.out1.GetData())
	copy(d.s2.GetData(), d.out2.GetData())
}
