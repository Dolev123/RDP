package com.dji.RDP.JavaCode;

//android imports

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;

import dji.common.camera.SettingsDefinitions;
import dji.common.error.DJIError;
import dji.common.flightcontroller.FlightControllerState;
import dji.common.gimbal.Rotation;
import dji.common.gimbal.RotationMode;
import dji.common.util.CommonCallbacks;
import dji.sdk.camera.Camera;
import dji.sdk.camera.DownloadListener;
import dji.sdk.camera.FetchMediaTask;
import dji.sdk.camera.FetchMediaTaskContent;
import dji.sdk.camera.FetchMediaTaskScheduler;
import dji.sdk.camera.MediaFile;
import dji.sdk.camera.MediaManager;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.mobilerc.MobileRemoteController;
import dji.sdk.products.Aircraft;

/**
 * A class to handle all connections with the aircraft from the DJI SDK.
 * has basic movement and control on the drone.
 *
 * */
public class TestClass {

    private static final String TAG = TestClass.class.getName();

     // the Aircraft class is a DJI class used to control the drone and it's components
    private Aircraft aircraft;
    public Handler mHandler;
    //the FlightController class is a DJI class that directly controls the drone and macro functions like takeoff and landing.
    private FlightController fc;
    //is the drone in air (flying)
    private boolean isAirborn = false;
    //the MobileRemoteController class is used to control the drones movement in all 3 axis in real-time from a connected mobile device.
    private MobileRemoteController mrc;
    public float height = 0;
    //the controlMode variable sets the output of the control function, if it controls the drone's movement or the gimbal's rotation.
    private boolean controlMode = false;
    //The camera on the aircraft
    private Camera camera;
    //media manager to access the photos/image files
    private MediaManager mediaManager;
    //list of all media files in
    private List<MediaFile> mediaFileList;
    //state of the file list in the SD
    private MediaManager.FileListState currentFileListState;
    //class that can schedule tasks to operate, used foe fetching media from the drome
    private FetchMediaTaskScheduler scheduler;
    //contains the paths to all pictures downloaded from the drone
    private Vector<String> paths;
    //is the gimbal facing down
    private boolean isGimbalDown = false;

    private Activity context;
    TestClass(Activity context) {
        this.context = context;
        TestClass2();
    }

    private void tv(String s) {
        Toast.makeText(context, "III "+s, Toast.LENGTH_SHORT).show();
    }

    public void TestClass2()
    {
        Log.d(TAG, "TestClass: Created");

        //initiating Aircraft as the product connected
        aircraft = (Aircraft) FPVDemoApplication.getProductInstance();
        fc = aircraft.getFlightController();
        mrc  = aircraft.getMobileRemoteController();
        camera = FPVDemoApplication.getCameraInstance();
        mediaFileList = new ArrayList<>();
        currentFileListState = MediaManager.FileListState.UNKNOWN;
        paths = new Vector<String>();

        //checks if downloading media is possible
        if (camera.isMediaDownloadModeSupported())
        {
            mediaManager = camera.getMediaManager();
            //checks if media manager is not null
            if(mediaManager != null) {
                scheduler = mediaManager.getScheduler();
            }
            else {
                scheduler = null;
            }
        }
        //when impossible Media Manager has no use
        else {
            Log.d("TestClass: ", "ERROR: TestClass: Media Download is not supported");
            mediaManager = null;
            scheduler = null;
        }

        //setting Aircraft's current location as Home
        fc.setHomeLocationUsingAircraftCurrentLocation(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onResult(DJIError djiError) {
                if(djiError != null) {
                    //setResultToToast(djiError.getDescription());
                    Log.d(TAG, "onResult: " + djiError.getDescription());
                }
            }
        });

        //something with logging. probably not needed
        aircraft.getFlightController().setStateCallback( new FlightControllerState.Callback() {

            /**
             * this function updates the UI to show the real time height and coordinates of the drone using the handler.
             * @param flightControllerState - the object which contains all the light stae variables that the app shows to the user like height and coordinates.
             */
            @Override
            public void onUpdate(@NonNull FlightControllerState flightControllerState) {
                try {

                } catch (Exception e) {
                    //Toast.makeText(MainActivity.this, "ERROR: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    Log.d("fcs.callback", "ERROR: " + e.getMessage());
                }
            }
        });

        Log.d(TAG, "TestClass: Functions");
    }

//////////////////////////BASIC FLIGHT START////////////////////////////////////////////////////////
    /**
     * the drone will take off.
     */
    public void takeOff() {
        //if connected and control flight
        if (fc.isConnected() && !controlMode) {
            fc.startTakeoff(new CommonCallbacks.CompletionCallback() {
                @Override
                public void onResult(DJIError djiError) {
                    if (djiError != null) {
                        Log.d("startTakeOff: ", "ERROR: TestClass: TakeOff: " + djiError.getDescription());
                    }
                    //when successful change state to in air
                    else {
                        isAirborn = true;
                    }
                }
            });
        }
    }

    /**
     * immediately stops the drone in midair.
     */
    public void stop () {
        //if connected
        if (fc.isConnected()) {
            //if in air
            if (isAirborn || true) {
                //set all joysticks to 0
                Log.d(TAG, "TestClass: stop: " + aircraft.getMobileRemoteController());
                mrc = aircraft.getMobileRemoteController();
                mrc.setLeftStickHorizontal(0);
                mrc.setLeftStickVertical(0);
                mrc.setRightStickHorizontal(0);
                mrc.setRightStickVertical(0);

                Log.d(TAG, "TestClass: stop: " + mrc.getLeftStickHorizontal() + mrc.getLeftStickVertical() + mrc.getRightStickHorizontal() + mrc.getRightStickVertical());
            }
        }
    }

    /**
     * the drone will land.
     */
    public void startLanding(){
        //if connected and control flight
        if (fc.isConnected() && !controlMode) {
            //stop();
            fc.startLanding(new CommonCallbacks.CompletionCallback() {
                @Override
                public void onResult(DJIError djiError) {
                    if (djiError != null) {
                        Log.d("startLanding: ", "ERROR: TestClass: startLanding: " + djiError.getDescription() + " " + djiError.toString());
                    }
                    //when successful change state to not in air
                    else {
                        isAirborn = false;
                    }
                }
            });
        }
    }
//////////////////////////BASIC FLIGHT END//////////////////////////////////////////////////////////

//////////////////////////CAMERA START//////////////////////////////////////////////////////////////
    /**
     * take picture on drone
     * */
    public void takePicture()
    {
        //setting camera mode to shoot photos
        camera.setMode(SettingsDefinitions.CameraMode.SHOOT_PHOTO, new CommonCallbacks.CompletionCallback(){
            @Override
            public void onResult(DJIError mError) {
                if (mError != null){
                    tv("In camera Set error");
                    Log.d(TAG, "ERROR: TestClass: Set Shoot Photo Mode Failed: " + mError.getDescription());
                }
                //Mode set was successful
                else {
                    tv("In camera Set succes");
                    Log.d(TAG, "takePicture: Mode Set");
                    //try to take picture
                    camera.startShootPhoto(new CommonCallbacks.CompletionCallback() {
                        @Override
                        public void onResult(DJIError djiError) {

                            if (djiError != null) {
                                tv("In camera shoot error");
                                Log.d(TAG, "ERROR: TestClass: Shoot Photo Failed: " + djiError.getDescription());
                            }
                            //photo successfully taken
                            else {
                                tv("In camera shoot success");

                                Log.d(TAG, "TestClass: takePicture: Photo taken");
                            }
                        }
                    });
                }
            }
        });
    }

    /**
     * takes a high resolution picture on drone and downloads it
     * */
    public void takeHighResolutionPicture()
    {
        //setting camera mode to shoot photos
        camera.setMode(SettingsDefinitions.CameraMode.SHOOT_PHOTO, new CommonCallbacks.CompletionCallback(){
            @Override
            public void onResult(DJIError mError) {
                if (mError != null){
                    Log.d(TAG, "ERROR: TestClass: Set Shoot Photo Mode Failed: " + mError.getDescription());
                }
                //Mode set was successful
                else {
                    Log.d(TAG, "takePicture: Mode Set");
                    //try to take picture
                    camera.setMediaFileCallback(new MediaFile.Callback() {
                        @Override
                        public void onNewFile(@NonNull MediaFile mediaFile) {
                            //path to storage
                            String path = Environment.getExternalStorageDirectory().toString();
                            //output stream to create the new file
                            OutputStream fOut;
                            //the file
                            File sFilse = new File(path, mediaFile.getFileName());

                            /*
                            fetching high resolution photo from the SD card to the phone
                             */
                            mediaFile.fetchFileData(sFilse, null, new DownloadListener<String>() {
                                @Override
                                public void onFailure(DJIError error) {
                                    Log.d(TAG, "TestClass: takeHighResolutionPicture:  fetchFileData: onFailure: " + error);
                                }

                                @Override
                                public void onProgress(long total, long current) {
                                }

                                @Override
                                public void onRateUpdate(long total, long current, long persize) {
                                }

                                @Override
                                public void onStart() {
                                }

                                @Override
                                public void onSuccess(String filePath) {
                                    //adding the path to paths
                                    Log.d(TAG, "TestClass: takeHighResolutionPicture:  fetchFileData: onSuccess: path is - " + filePath);
                                    paths.add(filePath);
                                }
                            });
                        }
                    });
                    camera.startShootPhoto(new CommonCallbacks.CompletionCallback() {
                        @Override
                        public void onResult(DJIError djiError) {
                            if (djiError != null) {
                                Log.d(TAG, "ERROR: TestClass: takeHighResolutionPicture: Shoot Photo Failed: " + djiError.getDescription());
                            }
                            //photo successfully taken
                            else {
                                Log.d(TAG, "TestClass: takeHighResolutionPicture: Photo taken");
                            }
                        }
                    });
                }
            }
        });
    }
//////////////////////////CAMERA END////////////////////////////////////////////////////////////////

//////////////////////////FILES HANDLE START////////////////////////////////////////////////////////

    /**
     * getter for paths variable
     * @return vector of paths to pictures
     */
    public Vector<String> getPaths()
    {
        return paths;
    }

    /**
     * formatting the SD card on the drone
     */
    public void Format()
    {
        if (camera != null) {
        camera.formatSDCard(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onResult(DJIError djiError) {
                if (djiError != null)
                {
                    Log.d(TAG, "ERROR: TestClass: Format: Formatting SD card Failed: " + djiError.getDescription());
                }
            }
        });
        }
    }

    public void downloadPreviews(final DownloadOptions dop)
    {
        downloadPreviews(dop, 0);
    }

    /**
     * downloads the photos from the drone to the phone
     */
    public void downloadPreviews(final DownloadOptions dop, final int index)
    {
        //if media manager is not null
        if (mediaManager != null)
        {
            //if media manager already busy, can't download
            if ((currentFileListState == MediaManager.FileListState.SYNCING) || (currentFileListState == MediaManager.FileListState.DELETING)){
                Log.d(TAG, "Media Manager is busy.");
            }else {
                //refreshing the file list of the media manager
                mediaManager.refreshFileList(new CommonCallbacks.CompletionCallback(){
                    @Override
                    public void onResult(DJIError djiError) {
                        if (djiError != null)
                        {
                            Log.d(TAG, "TestClass: downloadPreviews: RefreshFilter: onResult: ERROR: " + djiError.toString());
                        } else {
                            //Reset data
                            if (currentFileListState != MediaManager.FileListState.INCOMPLETE) {
                                mediaFileList.clear();
                            }
                            //get files (more or less)
                            mediaFileList = mediaManager.getFileListSnapshot();
                            //sorting files from recent to oldest
                            Collections.sort(mediaFileList, new Comparator<MediaFile>() {
                                @Override
                                public int compare(MediaFile lhs, MediaFile rhs) {
                                    if (lhs.getTimeCreated() < rhs.getTimeCreated()) {
                                        return 1;
                                    } else if (lhs.getTimeCreated() > rhs.getTimeCreated()) {
                                        return -1;
                                    }
                                    return 0;
                                }
                            });
                            //execute tasks in the scheduler
                            scheduler.resume(new CommonCallbacks.CompletionCallback() {
                                @Override
                                public void onResult(DJIError error) {
                                    if (error == null) {
                                       switch (dop)
                                       {
                                           case Full:
                                               getPreviews();
                                               break;
                                           case First:
                                               getPreviewByIndex(0);
                                               break;
                                           case OnePic:
                                               getPreviewByIndex(index);
                                               break;
                                           default:
                                               Log.d(TAG, "ERROR: TestClass: onResult: ");
                                               break;
                                       }
                                    } else {
                                        Log.d(TAG, "ERROR: TestClass: onResult: can't download");
                                    }
                                }
                            });
                        }
                    }
                });
            }
        }
    }

    /**
     * getting a specific thumbnail from the scheduler
     * @param index the index of the thumbnail to get with scheduler
     */
    private void getThumbnailByIndex(final int index) {
        FetchMediaTask task = new FetchMediaTask(mediaFileList.get(index), FetchMediaTaskContent.THUMBNAIL, taskCallback);
        scheduler.moveTaskToEnd(task);
    }

    /**
     * getting a specific preview from the scheduler
     * @param index the index of the preview to get with scheduler
     */
    private void getPreviewByIndex(final int index) {
        FetchMediaTask task = new FetchMediaTask(mediaFileList.get(index), FetchMediaTaskContent.PREVIEW, taskCallback);
        scheduler.moveTaskToEnd(task);
    }

    /**
     * getting all thumbnails
     */
    private void getThumbnails() {
        if (mediaFileList.size() <= 0) {
            Log.d(TAG,"ERROR: TestClass: No File info for downloading thumbnails");
            return;
        }
        for (int i = 0; i < mediaFileList.size(); i++) {
            getThumbnailByIndex(i);
        }
    }

    /**
     * getting all previews
     */
    private void getPreviews() {
        if (mediaFileList.size() <= 0) {
            Log.d(TAG,"ERROR: TestClass: No File info for downloading previews");
            return;
        }
        for (int i = 0; i < mediaFileList.size(); i++) {
            getPreviewByIndex(i);
        }
    }

    /**
     * task to scheduler that downloads a file to the phone.
     * it is a callback function variable
     */
    private FetchMediaTask.Callback taskCallback = new FetchMediaTask.Callback() {
        @Override
        public void onUpdate(MediaFile file, FetchMediaTaskContent option, DJIError error) {
            if (null == error) {

                //path to storage
                String path = Environment.getExternalStorageDirectory().toString();
                //output stream to create the new file
                OutputStream fOut;
                //the file
                File sFilse = new File(path, file.getFileName());

                //using try catch to prevent errors
                try {
                    fOut = new FileOutputStream(sFilse);
                    //convert to bitmap
                    Bitmap pictureBitmap = (file.getPreview());
                    //decide and save in a requested format (JPEG) with 85% compression
                    pictureBitmap.compress(Bitmap.CompressFormat.JPEG, 85, fOut);
                    //emptying and closing he stream
                    fOut.flush();
                    fOut.close();
                } catch (FileNotFoundException e) {
                    Log.d(TAG, "EXCEPTION: TestClass: taskCallback: " + e.toString());
                } catch (IOException e) {
                    Log.d(TAG, "EXCEPTION: TestClass: taskCallback: " + e.toString());
                }

                //adding the path to paths
                paths.add(path + "\\" + file.getFileName());

                //if the media wanted is preview
                if (option == FetchMediaTaskContent.PREVIEW) {
                    Log.d(TAG, "TestClass: PREVIEW path: "  + path + "\\" +file.getFileName());

                }
                //if the media wanted is thumbnail
                if (option == FetchMediaTaskContent.THUMBNAIL) {
                    Log.d(TAG, "TestClass: THUMBNAIL"+ file.getThumbnailCachePath() + "\\" +file.getFileName());
                }
            } else {
                Log.d(TAG, "ERROR: TestClass: taskCallback: Fetch Media Task Failed" + error.getDescription());
            }
        }
    };

//////////////////////////FILES HANDLE END//////////////////////////////////////////////////////////

//////////////////////////GIMBAL START//////////////////////////////////////////////////////////////

    public boolean GimbalDown() { return isGimbalDown; }

    /**
     * changes the gimbal's rotation between looking down and looking straight.
     */
    public void ChangeGimbal()
    {
        //check for errors and
        if (!aircraft.isConnected() || aircraft.getGimbal() == null)
        {
            return;
        }
        //creating the rotation
        float angle = isGimbalDown ? 0 : -75;
        Rotation rotation =new Rotation.Builder().pitch(angle)
                .mode(RotationMode.ABSOLUTE_ANGLE)
                .yaw(Rotation.NO_ROTATION)
                .roll(Rotation.NO_ROTATION)
                .time(0)
                .build();
        //rotates the gimbal to face down
        aircraft.getGimbal().rotate(rotation, new CommonCallbacks.CompletionCallback() {
            @Override
            public void onResult(DJIError djiError) {
                if (djiError != null)
                {
                    Log.d(TAG, "TestClass: ChangeGimbal: onResult: ERROR: " + djiError);
                }
                else {
                    isGimbalDown = !isGimbalDown;
                }
            }
        });


    }

//////////////////////////GIMBAL END////////////////////////////////////////////////////////////////

//////////////////////////BASIC MOVEMENT START//////////////////////////////////////////////////////

    /**
     * moves the drone in a wanted direction from (x,y,z) axises
     * @param d - direction of the wanted movement
     * */
    public void TestMove(Direction d)
    {
        //if not connected or not in air
        if (!fc.isConnected() || !isAirborn || true)
        {
            return;
        }
        //stop movement for drone
        stop();
        mrc = aircraft.getMobileRemoteController();
        //switch for direction given
        switch (d)
        {
            case UP:
                mrc.setLeftStickVertical((float) 0.5);
                break;
            case DOWN:
                mrc.setLeftStickVertical((float) -0.5);
                break;
            case LEFT:
                mrc.setRightStickHorizontal((float) -0.5);
                break;
            case RIGHT:
                mrc.setRightStickHorizontal((float) 0.5);
                break;
            case FORWARD:
                mrc.setRightStickVertical((float) 0.5);
                break;
            case BACKWARD:
                mrc.setRightStickVertical((float) -0.5);
                break;
            default:
                Log.d(TAG, "TestClass: TestMove: WTF: " + d.toString() + " should not exist");
                break;
        }
        Log.d(TAG, "TestClass: TestMove: " + mrc.getLeftStickHorizontal() + mrc.getLeftStickVertical() + mrc.getRightStickHorizontal() + mrc.getRightStickVertical());
    }
//////////////////////////BASIC MOVEMENT END////////////////////////////////////////////////////////
}

/**
 * enum for all sides of movement
 * */
enum Direction{
    FORWARD,
    BACKWARD,
    RIGHT,
    LEFT,
    UP,
    DOWN
}

/**
 * enum for download options
 */
enum DownloadOptions{
  OnePic,
  Full,
  First
}
