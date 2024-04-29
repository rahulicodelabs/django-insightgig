# example/views.py
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime, timedelta
from django.http import HttpResponse, JsonResponse
import os
import json
import requests
from api.settings import env

from example.apps_helper_function.transcribe_file import (
  process_audio_file,
  num_tokens_from_string,
  format_with_gpt4,
  save_transcription_to_word
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
  now = datetime.now()
  html = f'''
  <html>
      <body>
          <h1>Hello! App running successfully</h1>
          <p>The current time is { now }.</p>
      </body>
  </html>
  '''
  return HttpResponse(html)

# Upload the file aws S3 bucket  
def upload_to_s3(file_path, s3_file_name):
  
  # AWS credentials
  aws_access_key = env('AWS_S3_ACCESS_ID')
  aws_secret_key = env('AWS_SECRET_ACCESS_KEY')
  bucket_name = env('AWS_S3_BUCKET_NAME')
  
  # Create an S3 client
  s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
  # Get the current date and time + Add 15 days to the current date
  expiration_date = datetime.now() + timedelta(days=15)
  
  file_extension = os.path.splitext(s3_file_name)[1].lower()
  if file_extension == '.docx':
    content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  else:
    content_type = 'application/vnd.ms-excel'
    
  try:
    # Upload the file
    s3.upload_file(
      file_path, 
      bucket_name, 
      s3_file_name,
      ExtraArgs={
        'ContentType': content_type,
        'ACL': "public-read",
        'ContentDisposition': 'inline',
        'Expires': expiration_date,
      }
    )
    print("File uploaded successfully.")
    # Get the URL of the uploaded file
    s3_file_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_name}"
    return s3_file_url
  except FileNotFoundError:
    print("The file was not found.")
    return False
  except NoCredentialsError:
    print("Credentials not available.")
    return False
  
# Send file link to the email
def send_email(params):
  # Extracts email_to, link, app_name, and email_template from a dictionary and assigns them to variables.
  email_to = params.get("email_to")
  link = params.get("link")
  app_name = params.get("app_name")
  email_template = params.get("email_template")
  username = params.get("username")
  url = "https://api.zeptomail.in/v1.1/email/template"
  
  # ZeptoMail credentials
  api_token = env('ZEPTO_API_TOKEN')
  delivery_email = env('ZEPTO_EMAIL_FROM')
  delivery_name = env('ZEPTO_EMAIL_FROM_NAME')
  email_template_id = env(email_template)

  try:
    headers = {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      'Authorization': api_token,
    }

    payload = json.dumps({
      "mail_template_key": email_template_id,
      "from": {
        "address": delivery_email,
        "name": delivery_name
      },
      "to": [
        {
          "email_address": {
            "address": email_to,
            "name": "admin"
          }
        }
      ],
      "merge_info": {
        "appName": app_name,
        "link": link,
        "email": email_to,
        "username": username,
      }
    })

    requests.request("POST", url, headers=headers, data=payload)
    
  except Exception as e:
    print(e, "Error sending email!!")
    return e

# Transcribe file contents
def transcribe_file(request):
  if request.method != 'POST':
    return JsonResponse({'error': 'Unsupported method'}, status=405)
  
  try:
    # Extract file contents
    chunk = request.FILES['file']
    chunk_num = int(request.POST.get('chunk'))
    total_chunks = int(request.POST.get('totalChunks'))
    file_name = request.POST.get('fileName')
    
    # Extract other details
    input_helptext = request.POST.get('inputText')
    email = request.POST.get('email')
    username = request.POST.get('username')
    app_name = request.POST.get('appName')
    
    # Write the received chunk into file
    input_file_path = os.path.join(BASE_DIR, file_name)
    with open(input_file_path, 'ab') as f:
      for content in chunk.chunks():
        f.write(content)

    if chunk_num == total_chunks:
      # File upload complete - Perform any additional processing here
      print('file received',input_file_path)
  
      # Process the file and print the transcription.
      transcription = process_audio_file(input_file_path,input_helptext, delete_converted_mp3=True)
      
      # Count tokens using tiktoken.
      token_count = num_tokens_from_string(transcription)
      print(f"Token count for transcription: {token_count}")

      formatted_transcription = ""

      # # Check token count and split text if necessary.
      # if token_count < 3072:
      #   # Send the entire transcription to GPT-4 for formatting.
      #   formatted_transcription = format_with_gpt4([transcription], input_helptext)
       
      # else:
      #   # Split the transcription into chunks of 2048 tokens each and format each chunk.
      #   encoding = tiktoken.get_encoding("cl100k_base")
      #   tokens = encoding.encode(transcription)
      #   chunks = [encoding.decode(tokens[i:i + 2048]) for i in range(0, len(tokens), 2048)]
      #   formatted_transcription = format_with_gpt4(chunks, input_helptext) 
      #   logging.info("Formatting process completed.")
      #   print(formatted_transcription)
        
      # Remove the input file
      os.remove(input_file_path)
          
      if transcription.strip(): 
        output_file_path = os.path.join(BASE_DIR, "output.docx")
        # Generate docx file with the formatted transcript
        word_file_path = save_transcription_to_word(transcription, output_file_path)
        
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a dynamic filename with the timestamp
        dynamic_filename = f"output_{timestamp}.docx"

        # Use the dynamic filename in your function
        file_link = upload_to_s3(word_file_path, dynamic_filename)
        # Upload the file aws S3 bucket
        
        # Send file link to the email
        email_params = {
          "email_to": email,
          "username":username,
          "link": file_link,
          "app_name": app_name,
          "email_template":'ZEPTO_EMAIL_TEMPLATE_TRANSCRIPTION_APP_RESPONSE'
        }
        send_email(email_params)
        os.remove(word_file_path)

      # Print the final formatted transcription.
    return JsonResponse({
      'status': 200, 
      'message': 'Success',
      # 'data': formatted_transcription
    })
     
  except Exception as e:
    # Handle exceptions appropriately
    print(e, 'error')
    return JsonResponse({'error': f'Error processing the file: {str(e)}'}, status=400)