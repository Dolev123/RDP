package com.dji.RDP.JavaCode;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.PowerManager;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

//import com.EV3.Numeric.BluetoothEV3Service;
//import com.EV3.Numeric.DeviceListActivity;
//import com.EV3.Numeric.R;
/*
import android.view.inputmethod.EditorInfo;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ListView;
import android.view.KeyEvent;
*/

/**
 * This is the main Activity that displays the current session.
 * In this activity the connection to the robot is activated through button (which calls a connection activity)
 * Also in the activity all the main methods an abilities of the program will be available.
 */
public class EV3Numeric extends AppCompatActivity {

    // Debugging
    private static final String TAG = "EV3Numeric:NXTMailbox";
    private static final boolean DEBUG = true;

    private boolean drive = false;

    // Message types sent from the BluetoothEV3Service Handler
    public static final int MESSAGE_STATE_CHANGE = 1;
    public static final int MESSAGE_READ = 2;
    public static final int MESSAGE_WRITE = 3;
    public static final int MESSAGE_DEVICE_NAME = 4;
    public static final int MESSAGE_TOAST = 5;

    // Key names received from the BluetoothEV3Service Handler
    public static final String DEVICE_NAME = "device_name";
    public static final String TOAST = "toast";

    // Intent request codes
    private static final int REQUEST_CONNECT_DEVICE = 1;
    private static final int REQUEST_ENABLE_BT = 2;

    // Layout Views
    private TextView mTitle;
    private Button mSendButton0;
    private Button mConnectButton;


    // Name of the connected device
    private String mConnectedDeviceName = null;

    // Local Bluetooth adapter
    private BluetoothAdapter mBluetoothAdapter = null;

    // Member object for the EV3 services
    private BluetoothEV3Service mEV3Service = null;

    // WakeLock
    private PowerManager.WakeLock wl;

    private boolean abs = false;
    private Button btnABS;

    private Button btn_takeoff, btn_landing, btn_shoot, btn_stop, btn_download, btn_format;

    public static int direction = 0;

    //test class to control the aircraft
    private TestClass tc;

    //class for doing all the image processing
    private ImageProcessing imageProcessing;
    private ImageView ivProcessedImage;

    //path to the last downloaded picture/preview file
    private String currPath = "";

    /**
     * constructor for the activity.
     * @param savedInstanceState
     */
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        if (DEBUG) Log.e(TAG, "ON CREATE");

        //creating nrew object to controll the drone
        tc = new TestClass(this);

        // Set up the custom title
        mTitle = (TextView) findViewById(R.id.textView);
        //mTitle.setText(R.string.app_name);
        //mTitle = (TextView) findViewById(R.id.title_right_text);

        // Get local Bluetooth adapter
        mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();

        // If the adapter is null, then Bluetooth is not supported
        if (mBluetoothAdapter == null) {
            Msg("Bluetooth is not available");
            finish();
            return;
        }

        ivProcessedImage = (ImageView) findViewById(R.id.ivProcessedImage);

        // Stop turn off
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wl = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, TAG);


    }


    /**
     * starts when activity visible to user.
     * establishes BlueTooth connection
     */
    @Override
    public void onStart() {
        super.onStart();
        if (DEBUG) Log.e(TAG, "ON START");

        // If BT is not on, request that it be enabled.
        // setupApp() will then be called during onActivityResult
        if (!mBluetoothAdapter.isEnabled()) {
            Intent enableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            startActivityForResult(enableIntent, REQUEST_ENABLE_BT);
            // Otherwise, setup the session
        } else {
            if (mEV3Service == null) {
                setupApp();
            }
        }
    }

    /**
     * sends int arrayList to a robot on a specific mailbox
     *
     * @param arr     array to send
     * @param mailbox mailbox
     */
    public void send(ArrayList<Integer> arr, String mailbox) {
        while (!arr.isEmpty()) {
            mEV3Service.EV3.send(mailbox, arr.remove(0));
        }
    }

    /**
     * turn the array-list of the path given by dijkstra and converts it into commands for the EV3 robot
     * @param arr
     * @return
     * @throws Exception
     */
    public ArrayList<Integer> turnIntoCmds(ArrayList<xy> arr) throws Exception
    {
        ArrayList<Integer> ans=new ArrayList<Integer>();
        direction=0;
        xy p=arr.remove(0);
        while(!arr.isEmpty())
        {
            ans.add(p.direction(arr.get(0)));
            ans.add(p.distance(arr.get(0))*33);
            p=arr.remove(0);
        }
        ans.add(0);
        return ans;
    }

    /**
     * initiating UI Elements
     */
    private void setupApp() {
        if (DEBUG) Log.d(TAG, "setup");


        // Initialize the send button with a listener that for click events

        // Initializing the "Start driving!" button.
        Button startButton = (Button) findViewById(R.id.start);
        //Register an OnClickListener
        startButton.setOnClickListener(new OnClickListener() {

            public void onClick(View v) {
               /* // Toggle between "Start driving!" and "Stop driving!".
                if (mEV3Service.getState() != BluetoothEV3Service.STATE_CONNECTED) {
                    Msg(R.string.title_not_connected);
                    setResultToToast("not connected");
                    return;
                } */

                Log.d(TAG, "CHECK: 1");
                boolean canStart = tc.getPaths().size() > 0;

                //start if path of downloaded picture is found
                if (canStart) {
                    Log.d(TAG, "CHECK: 2");
                    setResultToToast("path acquired");
                    ///START: image processing needs to be here

                    //TODO Image processing
                    currPath = tc.getPaths().get(tc.getPaths().size() - 1);
                    Log.d(TAG, "CHECK: 2.1");
                    imageProcessing = new ImageProcessing(EV3Numeric.this, currPath);
                    Log.d(TAG, "CHECK: 2.2");
                    MyPoint[][] theMatrix= imageProcessing.colorThreshold();
                    Log.d(TAG, "CHECK: 2.3");
                    ivProcessedImage.setImageBitmap(imageProcessing.bm);
                    Log.d(TAG, "CHECK: 2.3");
                    ///END: image processing needs to be here
                    Log.d(TAG, "CHECK: 3.4");
                    //TODO activate Dijkstra
                    ArrayList<Circle> c = new ArrayList<>();
                    int endPointId = -1;
                    for (int i = 0; i < theMatrix.length; i++) {
                        for (int j = 0; j < theMatrix[i].length; j++) {
                            MyPoint p = theMatrix[i][j];
                           /* ArrayList<Integer> ne.ighbors = new ArrayList<>();
                            neighbors.add()*/
                           int id = i*theMatrix.length+j;
                           if(p.color.equals(Color.GREEN))
                               endPointId = id;
                            c.add(new Circle(id, p.x,p.y));
                        }
                    }
                    Log.d(TAG, "CHECK: 4");
                    c.get(0).findNeighbors(c);
                    ArrayList<Point> p = c.get(0).circleToPoint(c);
                    Graph g = new Graph(p.get(0),p.get(endPointId),p);
                    ArrayList<xy> route = g.getMinFin();
                    ArrayList<Integer> a1;
                    Log.d(TAG, "CHECK: 5");
                    try {
                        a1=turnIntoCmds(route);
                    } catch (Exception e) {
                        Toast.makeText(EV3Numeric.this,"Something is wrong",Toast.LENGTH_SHORT).show();
                        a1=null;
                    }
                    Log.d(TAG, "CHECK: 5");
                    //send(a1, "abc");
                    Log.d(TAG, "CHECK: 6");
                }
                //if no path is /downloaded yet, can't do image processing
                else {
                    setResultToToast("can't start yet");
                }



            }

        });


        mConnectButton = (Button) findViewById(R.id.ButtonC);
        mConnectButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {
                if (mEV3Service.getState() == BluetoothEV3Service.STATE_NONE)
                    startDeviceList();
            }
        });


        // Initialize the BluetoothEV3Service to perform bluetooth connections
        mEV3Service = new BluetoothEV3Service(this, mHandler);

        //initiating buttons for drone control
        btn_landing = (Button) findViewById(R.id.btn_landing);
        btn_takeoff = (Button) findViewById(R.id.btn_takeoff);
        btn_shoot = (Button) findViewById(R.id.btn_shoot);
        btn_download = (Button) findViewById(R.id.btn_download);

        //assigning function to when clicked

        //set button to drone takeoff
        btn_takeoff.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (view == btn_takeoff) {
                    tc.takeOff();
                    setResultToToast("Takeoff");
                }
            }
        });
        //sets button to drone landing
        btn_landing.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (view == btn_landing) {
                    tc.startLanding();
                    setResultToToast("Landing");
                }
            }
        });
        //sets button to take photo by drone
        btn_shoot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (view == btn_shoot) {
                    //makes sure the gimbal is looking down.
                    if (!tc.GimbalDown())
                    {
                        tc.ChangeGimbal();
                    }
                    tc.takePicture();
                    tc.downloadPreviews(DownloadOptions.First);
                    setResultToToast("taking picture");
                }
            }
        });
    }


    /**
     * when resuming to the activity
     */
    @Override
    public synchronized void onResume() {
        super.onResume();
        if (DEBUG) Log.e(TAG, "ON RESUME");
        wl.acquire();
        // Performing this check in onResume() covers the case in which BT was
        // not enabled during onStart(), so we were paused to enable it...
        // onResume() will be called when ACTION_REQUEST_ENABLE activity returns.
        if (mEV3Service != null) {
            // Only if the state is STATE_NONE, do we know that we haven't started already
            if (mEV3Service.getState() == BluetoothEV3Service.STATE_NONE) {
                // Start the Bluetooth services
                //  mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(
                //            Sensor.TYPE_ROTATION_VECTOR), mSensorManager.SENSOR_DELAY_NORMAL);


                mEV3Service.start();
            }
        }
    }

    @Override
    public synchronized void onPause() {
        super.onPause();

        if (DEBUG) Log.e(TAG, "ON PAUSE");
        wl.release();
    }

    @Override
    public void onStop() {
        super.onStop();

        if (DEBUG) Log.e(TAG, "ON STOP");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // Stop the Bluetooth services
        if (mEV3Service != null)
            mEV3Service.stop();
        drive = false;

        if (DEBUG) Log.e(TAG, "ON DESTROY");
    }

    // The Handler that gets information back from the BluetoothEV3Service
    @SuppressLint("HandlerLeak")
    private final Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MESSAGE_STATE_CHANGE:
                    if (DEBUG) Log.i(TAG, "MESSAGE_STATE_CHANGE: " + msg.arg1);
                    switch (msg.arg1) {
                        case BluetoothEV3Service.STATE_CONNECTED:
                            mTitle.setText(R.string.title_connected_to);
                            mTitle.append(mConnectedDeviceName);
                            break;
                        case BluetoothEV3Service.STATE_CONNECTING:
                            mTitle.setText(R.string.title_connecting);
                            break;
                        case BluetoothEV3Service.STATE_NONE:
                            mTitle.setText(R.string.title_not_connected);
                            break;
                    }
                    break;
                case MESSAGE_WRITE:
                    break;
                case MESSAGE_READ:

                    byte[] readBuf = (byte[]) msg.obj;
                    //byte[] bytes = (msg.obj).toString().getBytes();

                    try {
                        String s = (new String(readBuf, "UTF-8")).substring(13);//+ " | "+ new String(bytes,"UTF-8");

                    } catch (UnsupportedEncodingException e) {
                        e.printStackTrace();
                    }
                    break;
                // break;
                case MESSAGE_DEVICE_NAME:
                    // save the connected device's name
                    mConnectedDeviceName = msg.getData().getString(DEVICE_NAME);
                    Msg("Connected to " + mConnectedDeviceName);
                    break;
                case MESSAGE_TOAST:
                    Msg(msg.getData().getString(TOAST));
                    break;
            }
        }
    };

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (DEBUG) Log.d(TAG, "onActivityResult " + resultCode);
        switch (requestCode) {
            case REQUEST_CONNECT_DEVICE:
                // When DeviceListActivity returns with a device to connect
                if (resultCode == Activity.RESULT_OK) {
                    // Get the device MAC address
                    String address = data.getExtras()
                            .getString(DeviceListActivity.EXTRA_DEVICE_ADDRESS);
                    // Get the BLuetoothDevice object
                    BluetoothDevice device = mBluetoothAdapter.getRemoteDevice(address);
                    // Attempt to connect to the device
                    mEV3Service.connect(device);
                }
                break;
            case REQUEST_ENABLE_BT:
                // When the request to enable Bluetooth returns
                if (resultCode == Activity.RESULT_OK) {
                    // Bluetooth is now enabled, so set up a session
                    setupApp();
                } else {
                    // User did not enable Bluetooth or an error occured
                    Log.d(TAG, "BT not enabled");
                    Msg(R.string.bt_not_enabled_leaving);
                    finish();
                }
        }
    }

    private void startDeviceList() {
        Intent serverIntent = new Intent(this, DeviceListActivity.class);
        startActivityForResult(serverIntent, REQUEST_CONNECT_DEVICE);
    }

////////////////showing messages using the TOAST////////////////////////////////////////////////////

    private void Msg(String m) {
        Toast.makeText(this, m, Toast.LENGTH_LONG).show();
    }

    private void Msg(int m) {
        Toast.makeText(this, m, Toast.LENGTH_SHORT).show();
    }


    public void setResultToToast(final String string) {
        EV3Numeric.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(EV3Numeric.this, string, Toast.LENGTH_SHORT).show();
            }
        });
    }
////////////////////////////////////////////////////////////////////////////////////////////////////
}


 