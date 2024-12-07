import time
import discord
import speech_recognition as sr
from discord.ext import commands
import wave
import asyncio
from openai import AsyncOpenAI  
from pydub import AudioSegment
import os
from notion_client import Client
from datetime import datetime
import os
from dotenv import load_dotenv
import pyaudio
import threading
import signal
import sys
import re

# .env 파일 로드
load_dotenv()

# API 키 가져오기
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOTION_TOKEN = os.getenv('NOTION_TOKEN')
NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')

# OpenAI API 키 설정
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
# Notion 클라이언트 초기화
notion = Client(auth=NOTION_TOKEN)

# 회의록 템플릿
MEETING_TEMPLATE = """
# {title}

## 회의 정보
- 날짜: {date}
- 시간: {time}
- 참석자: {attendees}
- 장소: {channel_name}

## 주요 안건
{agenda}

## 논의 내용
{discussion}

## 주요 결정사항
{decisions}

## 후속 조치
{action_items}

## 다음 회의
- 날짜: {next_meeting_date}
- 안건: {next_meeting_agenda}
"""
def preprocess_text(text):
    """한국어 회의 텍스트 전처리"""
    replacements = {
        # 일반적인 인사말/맺음말
        r'안녕하세요|안녕하십니까|감사합니다|수고하세요|수고하셨습니다': '',
        
        # 회의 진행 관련 일반적 표현
        r'\(침묵\)|\(조용\)|\(잠시\)|\(웃음\)|\(박수\)': '',
        r'네네|네 네|음음|음 음|아 네|그 그': '',
        
        # 일반적인 불필요 표현
        r'어떻게 생각하시나요\?|어떻게 생각하십니까\?': '?',
        r'그러니까|그니까|그래서': '',
        
        # 반복되는 표현
        r'(네 )+': '네 ',
        r'(음 )+': '음 ',
        
        # 불필요한 공백과 줄바꿈 정리
        r'\s+': ' ',
        r'\n\s*\n': '\n'
    }
    
    processed_text = text
    for pattern, replacement in replacements.items():
        processed_text = re.sub(pattern, replacement, processed_text)
    
    return processed_text.strip()
class AudioReceiver(discord.VoiceClient):
    def __init__(self, client: discord.Client, channel: discord.VoiceChannel):
        super().__init__(client, channel)
        self.recording = False
        self.frames = []
        
        # PyAudio 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.record_thread = None
    
    def start_recording(self):
        """녹음 시작"""
        if not self.recording:
            self.recording = True
            self.frames = []
            
            # 오디오 스트림 시작
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # 별도 스레드에서 녹음 실행
            self.record_thread = threading.Thread(target=self._record)
            self.record_thread.start()
    
    def _record(self):
        """녹음 실행 (별도 스레드)"""
        while self.recording:
            try:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break
    
    def stop_recording(self):
        """녹음 중지"""
        if self.recording:
            self.recording = False
            if self.record_thread:
                self.record_thread.join()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def write_to_wav(self, filename):
        """녹음 데이터를 WAV 파일로 저장"""
        if not self.frames:
            return False
            
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Error saving WAV file: {e}")
            return False
    
    async def disconnect(self):
        """연결 종료"""
        if self.recording:
            self.stop_recording()
        if hasattr(self, 'p'):
            self.p.terminate()
        await super().disconnect()

async def transcribe_audio(audio_file):
    """음성 파일을 텍스트로 변환 - 청크 단위로 처리"""
    recognizer = sr.Recognizer()
    try:
        # 오디오 파일을 여러 청크로 분할
        audio_data = AudioSegment.from_wav(audio_file)
        chunk_length = 30000  # 30초 단위로 분할
        chunks = [audio_data[i:i+chunk_length] for i in range(0, len(audio_data), chunk_length)]
        
        full_text = []
        for i, chunk in enumerate(chunks):
            # 임시 파일로 저장
            chunk_file = f"temp_chunk_{i}.wav"
            chunk.export(chunk_file, format="wav")
            
            # 음성 인식
            with sr.AudioFile(chunk_file) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio, language='ko-KR')
                    full_text.append(text)
                except sr.UnknownValueError:
                    print(f"Chunk {i}: Speech not recognized")
                except sr.RequestError as e:
                    print(f"Chunk {i}: Could not request results; {e}")
            
            # 임시 파일 삭제
            os.remove(chunk_file)
        joined_text = " ".join(full_text)
        return preprocess_text(joined_text)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

async def summarize_with_template(text):
    """GPT를 사용하여 회의 내용을 템플릿 형식으로 요약 - 청크 단위로 처리"""
    try:
        # 텍스트를 적절한 크기로 분할
        max_chunk_size = 4000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        all_summaries = []
        for chunk in chunks:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                    아래의 정확한 형식으로 회의 내용을 요약해주세요:
                    1. 주요 안건:
                    [안건 내용]

                    2. 논의 내용:
                    [논의된 내용]

                    3. 주요 결정사항:
                    [결정된 사항들]

                    4. 후속 조치:
                    [향후 조치사항]
                    """},
                    {"role": "user", "content": chunk}
                ]
            )
            all_summaries.append(response.choices[0].message.content)

        # 여러 요약본을 하나로 통합
        if len(all_summaries) > 1:
            final_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "여러 요약본을 동일한 형식으로 하나로 통합해주세요."},
                    {"role": "user", "content": "\n\n".join(all_summaries)}
                ]
            )
            summary = final_response.choices[0].message.content
        else:
            summary = all_summaries[0]

        # 결과 파싱
        result = {
            'agenda': '내용 없음',
            'discussion': '내용 없음',
            'decisions': '내용 없음',
            'action_items': '내용 없음'
        }
        
        for section in summary.split('\n\n'):
            section = section.strip()
            if '1. 주요 안건:' in section:
                result['agenda'] = section.split('1. 주요 안건:')[1].strip()
            elif '2. 논의 내용:' in section:
                result['discussion'] = section.split('2. 논의 내용:')[1].strip()
            elif '3. 주요 결정사항:' in section:
                result['decisions'] = section.split('3. 주요 결정사항:')[1].strip()
            elif '4. 후속 조치:' in section:
                result['action_items'] = section.split('4. 후속 조치:')[1].strip()

        return result
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


async def create_notion_page(notion_client, database_id, meeting_data):
    """Notion 페이지 생성"""
    new_page = {
        "parent": {"database_id": database_id},
        "properties": {
            "이름": {
                "title": [
                    {
                        "text": {
                            "content": meeting_data['title']
                        }
                    }
                ]
            },
            "이벤트 시간": {
                "date": {
                    "start": meeting_data['date']
                }
            },
            "유형": {
                "select": {
                    "name": "팀 주간 회의"
                }
            }
        },
        "children": [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "회의 정보"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"채널: {meeting_data['channel_name']}"}}]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "주요 안건"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": meeting_data['agenda']}}]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "논의 내용"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": meeting_data['discussion']}}]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "주요 결정사항"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": meeting_data['decisions']}}]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "후속 조치"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": meeting_data['action_items']}}]
                }
            },
            {
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"type": "text", "text": {"content": "전체 회의 내용"}}],
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": meeting_data['full_transcript']}}]
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    return notion_client.pages.create(**new_page)

class MeetingBot(commands.Bot):
    def __init__(self, notion_token, notion_database_id):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.voice_states = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.notion = Client(auth=notion_token)
        self.notion_database_id = notion_database_id
        self.current_meeting = None
        
    async def setup_hook(self):
        print("\n=== API 연결 테스트 시작 ===")
        
        # Discord 연결 테스트
        print("\n1. Discord 연결 테스트:")
        print(f"✓ 봇 이름: {self.user.name}")
        print(f"✓ 봇 ID: {self.user.id}")
        
        # OpenAI 연결 테스트
        print("\n2. OpenAI API 테스트:")
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "테스트"}],
                max_tokens=5
            )
            print("✓ OpenAI API 연결 성공")
        except Exception as e:
            print(f"✗ OpenAI API 연결 실패: {str(e)}")
        
        # Notion 연결 테스트
        print("\n3. Notion API 테스트:")
        try:
            self.notion.databases.retrieve(database_id=self.notion_database_id)
            print("✓ Notion API 연결 및 데이터베이스 접근 성공")
        except Exception as e:
            print(f"✗ Notion API 연결 실패: {str(e)}")
        
        print("\n=== API 연결 테스트 완료 ===")

# bot 인스턴스 생성
bot = MeetingBot(
    notion_token=NOTION_TOKEN,
    notion_database_id=NOTION_DATABASE_ID
)
@bot.command()
async def start(ctx, *, title: str = None, duration: int = 60):
    """회의 녹음 시작 (기본 60분)"""
    if ctx.author.voice is None:
        await ctx.send("음성 채널에 먼저 입장해주세요!")
        return
        
    if not title:
        await ctx.send("회의 제목을 입력해주세요. 예: !start 주간 팀 미팅")
        return
        
    try:
        voice_channel = ctx.author.voice.channel
        voice_client = await voice_channel.connect(cls=AudioReceiver)
        voice_client.start_recording()
        
        # 참석자 목록 생성
        attendees = [member.name for member in voice_channel.members]
        
        bot.current_meeting = {
            'title': title,
            'channel_name': voice_channel.name,
            'start_time': datetime.now(),
            'attendees': ', '.join(attendees)
        }
        
        await ctx.send(f"'{title}' 회의 녹음을 시작합니다.\n참석자: {', '.join(attendees)}\회의 녹음은 더 정확한 요약을 위해 {duration}분 후에 자동으로 종료됩니다.\n ")
        
        # 자동 종료 타이머 설정
        await asyncio.sleep(duration * 60)
        if ctx.voice_client:
            await ctx.invoke(bot.get_command('stop'))
            
    except Exception as e:
        await ctx.send(f"녹음 시작 중 오류가 발생했습니다: {str(e)}")
        if 'voice_client' in locals():
            await voice_client.disconnect()

@bot.command()
async def stop(ctx):
    """회의 녹음 종료 및 처리"""
    voice_client = ctx.guild.voice_client
    if voice_client and isinstance(voice_client, AudioReceiver):
        filename = f"meeting_{ctx.guild.id}_{ctx.channel.id}.wav"
        status_message = await ctx.send("처리 진행률:\n⬜⬜⬜⬜⬜ 0%")
        
        try:
            # 녹음 중지 및 파일 저장
            voice_client.stop_recording()
            if voice_client.write_to_wav(filename):
                await status_message.edit(content="처리 진행률:\n⬛⬜⬜⬜⬜ 20%")
                
                # 음성을 텍스트로 변환
                transcript = await transcribe_audio(filename)
                if transcript:
                    await status_message.edit(content="처리 진행률:\n⬛⬛⬜⬜⬜ 40%")
                    
                    # 텍스트 요약
                    summary = await summarize_with_template(transcript)
                    if summary:
                        await status_message.edit(content="처리 진행률:\n⬛⬛⬛⬜⬜ 60%")
                        
                        # 회의 데이터 구성
                        meeting_data = {
                            'title': bot.current_meeting['title'],
                            'date': bot.current_meeting['start_time'].strftime('%Y-%m-%d'),
                            'time': bot.current_meeting['start_time'].strftime('%H:%M'),
                            'channel_name': bot.current_meeting['channel_name'],
                            'attendees': bot.current_meeting['attendees'],
                            **summary,
                            'next_meeting_date': '',
                            'next_meeting_agenda': '',
                            'full_transcript': transcript
                        }
                        
                        await status_message.edit(content="처리 진행률:\n⬛⬛⬛⬛⬜ 80%")
                        
                        # Notion 페이지 생성
                        try:
                            page = await create_notion_page(bot.notion, bot.notion_database_id, meeting_data)
                            page_id = page["id"]
                            page_url = f"https://notion.so/{page_id.replace('-', '')}"
                            await status_message.edit(content="처리 진행률:\n⬛⬛⬛⬛⬛ 100%")
                            await ctx.send(f"회의록이 Notion에 저장되었습니다.\nURL: {page_url}")
                            
                            # 채널에 요약본 전송
                            formatted_summary = MEETING_TEMPLATE.format(**meeting_data)
                            await ctx.send("회의 요약:\n" + formatted_summary)
                        except Exception as e:
                            await ctx.send(f"Notion 저장 중 오류가 발생했습니다: {str(e)}")
                    else:
                        await ctx.send("요약 중 오류가 발생했습니다.")
                else:
                    await ctx.send("음성 인식 중 오류가 발생했습니다.")
            else:
                await ctx.send("녹음 파일 저장 중 오류가 발생했습니다.")
        except Exception as e:
            await ctx.send(f"처리 중 오류가 발생했습니다: {str(e)}")
        finally:
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Error removing temporary file: {e}")
            await voice_client.disconnect()
    else:
        await ctx.send("현재 진행 중인 녹음이 없습니다.")

@bot.command(name='guide')  
async def bot_guide(ctx):
    guide_text = """
**회의 녹음 봇 사용법**

**기본 명령어**
`!start [회의 제목]` - 회의 녹음 시작
예시: `!start 주간 팀 미팅`

`!stop` - 회의 녹음 종료 및 회의록 생성

**사용 순서**
1. 먼저 음성 채널에 입장하세요
2. `!start` 명령어로 회의 시작
3. 회의 진행
4. `!stop` 명령어로 회의 종료
5. 자동으로 회의록이 생성되고 Notion에 저장됩니다
"""
    await ctx.send(guide_text)

    

# test 명령어 추가
@bot.command(name='test')
@commands.is_owner()  # 봇 소유자만 실행 가능
async def test_connections(ctx):
    """API 연결 상태 테스트"""
    message = await ctx.send("🔍 API 연결 상태 확인 중...")
    
    results = []
    results.append("🤖 **API 연결 상태**")
    
    # Discord
    try:
        await bot.wait_until_ready()
        results.append("✅ Discord: 정상")
    except Exception as e:
        results.append(f"❌ Discord: {str(e)}")
    
    # OpenAI
    try:
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        results.append("✅ OpenAI: 정상")
    except Exception as e:
        results.append(f"❌ OpenAI: {str(e)}")
    
    # Notion
    try:
        bot.notion.users.list()
        results.append("✅ Notion: 정상")
    except Exception as e:
        results.append(f"❌ Notion: {str(e)}")
    
    await message.edit(content="\n".join(results))
@bot.command(name='apitest')
@commands.is_owner()
async def test_apis(ctx):
    """실시간 API 상태 확인"""
    status_msg = await ctx.send("🔄 API 상태 확인 중...")
    
    results = []
    results.append("📊 **API 상태 보고서**")
    
    # Discord 테스트
    results.append("\n**Discord**")
    results.append(f"✓ 봇 응답 시간: {round(bot.latency * 1000)}ms")
    
    # OpenAI 테스트
    results.append("\n**OpenAI**")
    try:
        start_time = time.time()
        await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        api_time = (time.time() - start_time) * 1000
        results.append(f"✓ API 응답 시간: {round(api_time)}ms")
    except Exception as e:
        results.append(f"✗ 연결 오류: {str(e)}")
    
    # Notion 테스트
    results.append("\n**Notion**")
    try:
        start_time = time.time()
        bot.notion.databases.retrieve(database_id=bot.notion_database_id)
        api_time = (time.time() - start_time) * 1000
        results.append(f"✓ API 응답 시간: {round(api_time)}ms")
    except Exception as e:
        results.append(f"✗ 연결 오류: {str(e)}")
    
    await status_msg.edit(content="\n".join(results))
def signal_handler(sig, frame):
    """프로그램 종료 시 정리 작업 수행"""
    print("\n프로그램을 종료합니다...")
    
    # 모든 음성 클라이언트 연결 해제
    for guild in bot.guilds:
        if guild.voice_client:
            asyncio.run(guild.voice_client.disconnect())
    
    # 봇 종료
    asyncio.run(bot.close())
    sys.exit(0)

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)

#Page 생성 Test용 코드
def create_page():
    try:
        # 테스트 페이지 생성
        new_page = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "이름": {
                    "title": [
                        {
                            "text": {"content": "테스트 회의록"}
                        }
                    ]
                },
                "이벤트 시간": {
                    "date": {
                        "start": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                },
                "유형": {
                    "select": {
                        "name": "팀 주간 회의"
                    }
                }
            },
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": "테스트 내용입니다."}}]
                    }
                }
            ]
        )
        
        # 페이지 ID로 URL 생성
        page_id = new_page["id"]
        page_url = f"https://notion.so/{page_id.replace('-', '')}"
        
        print("✅ 페이지 생성 성공!")
        print(f"생성된 페이지 URL: {page_url}")
        return page_url
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None
try:
    print("봇 시작 시도 중...")
    bot.run(DISCORD_TOKEN)
except Exception as e:
    print(f"봇 실행 중 오류 발생: {str(e)}")