package com.dji.RDP.JavaCode;



//import com.EV3.Numeric.BluetoothEV3Service;

public class EV3Brick {
        //the Bluetooth connection
	   BluetoothEV3Service os;

	   //c'tor
	   EV3Brick(BluetoothEV3Service mNxtService) {
	       os = mNxtService;
	   }


    /**
     * sends a number to the mailbox of the robot
     * @param n - the mailbox'es name (string)
     * @param b - number to send (float)
     */
	public void send( String n,float b){
		byte[] cmd = new byte[16+n.length()];//14 as the numbers of the indexed needed (look at the table char for messaging)
		int pos=0;
		//always working with pairs so 0+1 is 1 word...working with words and not bytes
		//cmd[0]+cmd[1] is the first word where cmd[0] is low 8 bits and cmd[1] is high 8 bits (we shift the length 8 times left to get the high 8 bits)
		cmd[0] = (byte)( (cmd.length - 2) & 0xFF );
		cmd[1] = (byte)( (cmd.length - 2) >> 8 );

		//cmd[2]+cmd[3] are message counter
		cmd[2] = (byte)0x00;
		cmd[3] = (byte)0x00;

		//Command type being a system command with no reply (0x81 = 129).
		cmd[4] = (byte)0x81;

		//System command number for WRITEMAILBOX (0x9E = 158).
		cmd[5] = (byte)0x9E;

		//Length of the mailbox name including zero termination character.
		cmd[6] = (byte)(n.length()+1);//message title length ,for example CompassX -> 8..
		pos = 7;
		for(int i = 0; i < n.length() && i < 255; i++ )
		{
			//Mailbox name that you want the message to be sent to e.g. this value corresponds to String n (Title)
			cmd[pos] = (byte)n.charAt(i);//Title described in UTF-8 encoding
			pos++;
		}

		//zero terminator
		cmd[pos] = (byte)0x00;
		pos++;

		//length of message (float) we choose it 4 bytes
		cmd[pos] = (byte)0x04;
		pos++;
		cmd[pos] = (byte)0x00;
		pos++;

		//Message or payload e.g 117.695
		//float b=(float)117.695;
		int bitsVal = Float.floatToIntBits(b);
		String padded = String.format("%32s", Integer.toBinaryString(bitsVal)).replace(' ', '0');
		String a1 = padded.substring(0,8);
		String b2 = padded.substring(8,16);
		String c3 = padded.substring(16,24);
		String d4 = padded.substring(24,32);


		int aa = Integer.parseInt(a1, 2); // MSB
		int bb = Integer.parseInt(b2, 2);
		int cc = Integer.parseInt(c3, 2);
		int dd = Integer.parseInt(d4, 2); //LSB


		cmd[pos] = (byte) dd;
		pos++;
		cmd[pos] = (byte) cc;
		pos++;
		cmd[pos] = (byte) bb;
		pos++;
		cmd[pos] = (byte) aa;

		os.write( cmd );

	}
}
