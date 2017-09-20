package ibm.interview.wesleykirinya.soundchek;

import android.Manifest;
import android.animation.Animator;
import android.animation.ObjectAnimator;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.TypedValue;
import android.view.View;
import android.view.animation.DecelerateInterpolator;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.TextView;

import org.apache.commons.math3.stat.regression.SimpleRegression;

import java.text.DecimalFormat;

public class AgeCheckActivity extends AppCompatActivity {

    private boolean playPauseBtnFirstPress = true;
    private AudioAnalysis audioAnalysis = null;
    private final int MY_PERMISSIONS_REQUEST_RECORD_AUDIO = 0x01;
    private CompoundButton.OnCheckedChangeListener onCheckedChangeListener =
            new CompoundButton.OnCheckedChangeListener() {
                @Override
                public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                    playPauseOnCheckedChanged(buttonView, isChecked);
                }
            };
    // For linear regression analysis (Machine learning)
    private SimpleRegression simpleRegression = new SimpleRegression(true);

    private class AudioAnalysis extends AsyncTask<Void, Double, Void> {
        public double getFrequency() {
            return frequency;
        }

        private double frequency = 0.0;

        @Override
        protected void onPreExecute() {
            TextView feedbackView = (TextView) findViewById(R.id.textView2);
            feedbackView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 60);
            feedbackView.setText("0 Hz");
        }

        @Override
        protected Void doInBackground(Void... params) {
            // Listen to mic and analyse audio

            final int samplesSz = 32768;
            final short[] samples = new short[samplesSz];
            final double[] fftReal = new double[samplesSz];
            final double[] fftImaginary = new double[samplesSz];
            final FFT fftHelper = new FFT(samplesSz);
            final double[] magnitudes = new double[samplesSz];

            final int sampleRate = 44100;
            int minBuffSz = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
            AudioRecord audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, minBuffSz);
            audioRecord.startRecording();

            do {
                int readCount = audioRecord.read(samples, 0, samplesSz);

                // Reset fftReal, fftImaginary
                for (int i = 0; i < samplesSz; ++i) {
                    fftReal[i] = fftImaginary[i] = 0.0;
                }

                // Normalize to -1.0 to 1.0
                for (int i = 0; i < samplesSz && i < readCount; ++i) {
                    fftReal[i] = (double) samples[i] / ((double) Short.MAX_VALUE);
                }

                fftHelper.fft(fftReal, fftImaginary);

                // Derive peak frequency
                for (int i = 0; i < samplesSz; ++i) {
                    magnitudes[i] = Math.sqrt((fftReal[i] * fftReal[i]) + (fftImaginary[i] * fftImaginary[i]));
                }
                double peakVal = -1.0;
                int pIndex = -1;
                for (int i = 0; i < samplesSz; ++i) {
                    if (peakVal < magnitudes[i]) {
                        peakVal = magnitudes[i];
                        pIndex = i;
                    }
                }

                double frequencyTmp = (sampleRate * pIndex) / samplesSz;
                if (frequencyTmp < 20 || frequencyTmp > 20000) {
                    // Ignore frequency beyond hearing range. Check why these are included
                    continue;
                }
                frequency = frequencyTmp;
                publishProgress(frequency);
            } while (!this.isCancelled());

            audioRecord.stop();
            audioRecord.release();

            return null;
        }

        protected void onProgressUpdate(Double... frequencies) {
            // Update UI
            DecimalFormat df = new DecimalFormat();
            df.setMaximumFractionDigits(0);
            String frequencyS = df.format(frequencies[0]) + " Hz";
            TextView feedbackView = (TextView) findViewById(R.id.textView2);
            feedbackView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 60);
            feedbackView.setText(frequencyS);
        }
    }

    private class FFT {
        int n, m, halfN;

        // Lookup tables (for optimisation)
        double[] cos;
        double[] sin;

        public FFT(int n1) {
            n = n1;
            halfN = n / 2;
            m = (int) (Math.log(n) / Math.log(2));

            // Check n is a power of 2
            if (n != (1 << m)) {
                throw new RuntimeException("FFT length must be power of 2");
            }

            // pre-compute tables
            cos = new double[halfN];
            sin = new double[halfN];

            for (int i = 0; i < halfN; i++) {
                cos[i] = Math.cos(-2 * Math.PI * i / n);
                sin[i] = Math.sin(-2 * Math.PI * i / n);
            }
        }

        /***************************************************************
         * fft.c
         * Douglas L. Jones
         * University of Illinois at Urbana-Champaign
         * January 19, 1992
         * http://cnx.rice.edu/content/m12016/latest/
         * <p/>
         * fft: in-place radix-2 DIT DFT of a complex input
         * <p/>
         * input:
         * n: length of FFT: must be a power of two
         * m: n = 2**m
         * input/output
         * x: double array of length n with real part of data
         * y: double array of length n with imag part of data
         * <p/>
         * Permission to copy and use this program is granted
         * as long as this header is included.
         ****************************************************************/
        public void fft(double[] x, double[] y) {
            int i, j, k, n1, n2, a;
            double c, s, t1, t2;

            // Bit-reverse
            j = 0;
            n2 = halfN;
            for (i = 1; i < n - 1; i++) {
                n1 = n2;
                while (j >= n1) {
                    j = j - n1;
                    n1 = n1 / 2;
                }
                j = j + n1;

                if (i < j) {
                    t1 = x[i];
                    x[i] = x[j];
                    x[j] = t1;
                    t1 = y[i];
                    y[i] = y[j];
                    y[j] = t1;
                }
            }

            // FFT
            n1 = 0;
            n2 = 1;

            for (i = 0; i < m; i++) {
                n1 = n2;
                n2 = n2 + n2;
                a = 0;

                for (j = 0; j < n1; j++) {
                    c = cos[a];
                    s = sin[a];
                    a += 1 << (m - i - 1);

                    for (k = j; k < n; k = k + n2) {
                        t1 = c * x[k + n1] - s * y[k + n1];
                        t2 = s * x[k + n1] + c * y[k + n1];
                        x[k + n1] = x[k] - t1;
                        y[k + n1] = y[k] - t2;
                        x[k] = x[k] + t1;
                        y[k] = y[k] + t2;
                    }
                }
            }
        }
    }


    /**
     * Helper interface to run custom code when UI animation ends
     */
    private interface AnimExt {
        public void onEnd();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_age_check);

        CheckBox playPauseBtn = (CheckBox) findViewById(R.id.checkBox);
        playPauseBtn.setOnCheckedChangeListener(onCheckedChangeListener);

        // Pre populate with known data. This is a set of data used to train the system where
        // age and hearing frequency loss is know for a subset of the population.
        simpleRegression.addData(new double[][]{
                {12000, 50},
                {15000, 40},
                {16000, 30},
                {17000, 24},
                {19000, 20}
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_RECORD_AUDIO: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    String text = getString(R.string.status_press_play);
                    TextView textView = (TextView) findViewById(R.id.textView);
                    textView.setText(text);
                } else {
                    CheckBox playPauseBtn = (CheckBox) findViewById(R.id.checkBox);
                    playPauseBtn.setOnCheckedChangeListener(null);
                    playPauseBtn.setChecked(false);
                    playPauseBtn.setOnCheckedChangeListener(onCheckedChangeListener);
                }
                return;
            }
        }
    }

    private void playPauseOnCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        boolean permSet = false;

        // Check permissions (Android 6+)
        if (ContextCompat.checkSelfPermission(AgeCheckActivity.this,
                Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(AgeCheckActivity.this,
                    Manifest.permission.RECORD_AUDIO)) {
                String text = getString(R.string.status_perm_needed_mic);
                TextView textView = (TextView) findViewById(R.id.textView);
                textView.setText(text);
            } else {
                ActivityCompat.requestPermissions(AgeCheckActivity.this,
                        new String[]{Manifest.permission.RECORD_AUDIO},
                        MY_PERMISSIONS_REQUEST_RECORD_AUDIO);
            }
        } else {
            permSet = true;
        }


        if (permSet) {
            if (isChecked && playPauseBtnFirstPress) {
                animateControls();
                playPauseBtnFirstPress = false;
            }

            if (isChecked) {
                // Setup and start audio analysis
                audioAnalysis = new AudioAnalysis();
                audioAnalysis.execute();
            } else {
                // Stop and tear down audio analysis
                audioAnalysis.cancel(true);

                // Use the current frequency value to determine age
                String age = age(audioAnalysis.getFrequency());
                String text = getString(R.string.status_age, age);
                TextView feedbackView = (TextView) findViewById(R.id.textView2);
                feedbackView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
                feedbackView.setText(text);
            }
        } else {
            buttonView.setOnCheckedChangeListener(null);
            buttonView.setChecked(false);
            buttonView.setOnCheckedChangeListener(onCheckedChangeListener);
        }
    }

    private String age(double frequency) {
        double age = simpleRegression.predict(frequency);
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(0);
        String ageDesc = df.format(age);

        return ageDesc;
    }

    private void animateControls() {
        CheckBox playPauseBtn = (CheckBox) findViewById(R.id.checkBox);
        verticalDisplace(playPauseBtn, 150, new AnimExt() {
            @Override
            public void onEnd() {
                TextView feedbackView = (TextView) findViewById(R.id.textView2);
                feedbackView.setVisibility(View.VISIBLE);
            }
        });

        TextView textView = (TextView) findViewById(R.id.textView);
        textView.setText(R.string.status_test_ongoing);
    }

    private void verticalDisplace(final View view, final int displacement, final AnimExt animExt) {
        ObjectAnimator animator = ObjectAnimator.ofFloat(view, "translationY", displacement);
        animator.setInterpolator(new DecelerateInterpolator());
        animator.setDuration(500);
        animator.addListener(new Animator.AnimatorListener() {
            @Override
            public void onAnimationStart(Animator animation) {
                // Empty
            }

            @Override
            public void onAnimationEnd(Animator animation) {
                if (animExt != null) {
                    animExt.onEnd();
                }
            }

            @Override
            public void onAnimationCancel(Animator animation) {
                // Empty
            }

            @Override
            public void onAnimationRepeat(Animator animation) {
                // Empty
            }
        });
        animator.start();
    }

}
