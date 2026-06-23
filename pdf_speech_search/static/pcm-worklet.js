class PcmWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions?.targetRate || 16000;
    this.sourceRate = sampleRate;
    this.ratio = this.sourceRate / this.targetRate;
    this.position = 0;
    this.pending = [];
    this.chunkSize = Math.round(this.targetRate * 0.08);
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) {
      return true;
    }

    const output = [];
    while (this.position < input.length) {
      const idx = Math.floor(this.position);
      const nextIdx = Math.min(idx + 1, input.length - 1);
      const frac = this.position - idx;
      const sample = input[idx] + (input[nextIdx] - input[idx]) * frac;
      output.push(Math.max(-1, Math.min(1, sample)));
      this.position += this.ratio;
    }
    this.position -= input.length;

    for (const sample of output) {
      this.pending.push(sample);
      if (this.pending.length >= this.chunkSize) {
        const pcm = new Int16Array(this.pending.length);
        for (let i = 0; i < this.pending.length; i += 1) {
          const value = this.pending[i];
          pcm[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this.pending = [];
      }
    }

    return true;
  }
}

registerProcessor("pcm-worklet", PcmWorklet);

