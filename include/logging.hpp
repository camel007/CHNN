#ifndef _CAFFE_LOGGING_H_
#define _CAFFE_LOGGING_H_

#include <assert.h>
#include <iostream>
#include <sstream>
#include <ctime>
#include <stdexcept>
namespace caffe{
	struct Error : public std::runtime_error
	{
		explicit Error(const std::string &s) : std::runtime_error(s) {}
	};
}

//#if defined(_MSC_VER) && _MSC_VER < 1900
//#define noexcept(a)
//#endif

#define CAFFE_THROW_EXCEPTION

namespace caffe{

	class LogMessageFatal
	{
	public:
		LogMessageFatal(const char* file, int line)
		{
			log_stream_ << file << ":" << line << ": ";
		}
		std::ostringstream &stream() { return log_stream_; }
		~LogMessageFatal()
		{
		}

	private:
		std::ostringstream log_stream_;
		LogMessageFatal(const LogMessageFatal&);
		void operator=(const LogMessageFatal);
	};

	class DateLogger {
	public:
		DateLogger() {
#if defined(_MSC_VER)
			_tzset();
#endif
		}
		const char* HumanDate() {
#if defined(_MSC_VER)
			_strtime_s(buffer_, sizeof(buffer_));
#else
			time_t time_value = time(NULL);
			struct tm *pnow;
#if !defined(_WIN32)
			struct tm now;
			pnow = localtime_r(&time_value, &now);
#else
			pnow = localtime(&time_value);  // NOLINT(*)
#endif
			snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d",
				pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif
			return buffer_;
		}

	private:
		char buffer_[9];
	};
}

#define CHECK(x)    \
if (!(x))			\
	caffe::LogMessageFatal(__FILE__, __LINE__).stream() << "Check " \
	"failed: " #x << ' '
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x) \
	((x) == NULL ? caffe::LogMessageFatal(__FILE__, __LINE__).stream() << "Check notnull: " #x << ' ', (x))
// Debug-only checking.
#ifdef NDEBUG
#define DCHECK(x) \
while (false) CHECK(x)
#define DCHECK_LT(x, y) \
while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
while (false) CHECK((x) != (y))
#else
#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))
#endif  // NDEBUG

#define LOG(__CONTENT__) \


#endif