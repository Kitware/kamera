//
// Created by Michael McDermott on 1/20/21.
//

#include <vector>
#include "roskv/envoy.h"
#include "roskv/archiver.h"


inline std::string isoformat_u(time_t tv_sec, long tv_nsec) {
    /// 20020131T100001,123456789
    /// old: glacial_2020/fl01/center_view/glacial_2020_fl01_C_20200901_212303.110970_meta.json
    /// stupid _ kludge because original format used underscore instead of ISO8601, and customer hates change

    char buf_sec[16];
    char buf_frac[16];
    strftime(buf_sec, sizeof(buf_sec), "%Y%m%d_%H%M%S", gmtime(&tv_sec));
    double tv_usec = ((double) tv_nsec) / 1e3; 
    int64_t int_usec = std::floor(tv_usec);
    double fracs = ((double) int_usec ) / 1e6;
    sprintf(buf_frac, "%0.6lf", fracs);
    return std::string(buf_sec) + std::string(buf_frac).substr(1);
}

Cameratype::Cameratype(const std::string &fov, const std::string &mode) :
        fov{fov}, mode{mode} {}

Cameratype Cameratype::from_env() {
    const char *env_fov = std::getenv("CAM_FOV");
    const char *env_mode = std::getenv("CAM_MODE");
    if (!(env_fov && env_fov[0])) {
        throw std::invalid_argument("ENV variable CAM_FOV must be specified");
    }
    if (!(env_mode && env_mode[0])) {
        throw std::invalid_argument("ENV variable CAM_MODE must be specified");
    }
    const std::string fov{env_fov};
    const std::string mode{env_mode};
    return Cameratype{fov, mode};
}


json::json FormatterHelper::conformKwargsToFormatter(const std::string &fmtr, const json::json &kwargs) {
    static const std::regex PAT_get_braced(R"(\{(\w+)\})");

    // ensure that all the necessary named _kwargs in the template are present in the dict
    auto m1 = std::sregex_iterator(fmtr.begin(), fmtr.end(), PAT_get_braced);
    auto m2 = std::sregex_iterator();
    json::json fmt_kwargs;

    for (auto el = m1; el != m2; ++el) {
        auto match = *el;
        std::string sval = match[1].str();  // capture group 1 = word inside braces
        bool has = kwargs.contains(sval);
//        fmt::print("{}: {}\n", sval, has);
        if (has) {
            fmt_kwargs[sval] = kwargs[sval];
        } else {
            fmt_kwargs[sval] = "(" + sval + ")";
        }
    }

    return fmt_kwargs;
}

std::string FormatterHelper::applyFormat(const std::string &fmtr, const json::json &kwargs, bool partial) {
    static const std::regex placeholder(R"(\{(\w+)\})");
    std::string out;
    std::size_t last_pos = 0;
    auto begin = std::sregex_iterator(fmtr.begin(), fmtr.end(), placeholder);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        const auto &match = *it;
        out += fmtr.substr(last_pos, match.position() - last_pos);
        std::string key = match[1].str();
        out += kwargs.contains(key) ? std::string(kwargs[key]) : match[0].str();
        last_pos = match.position() + match.length();
    }
    out += fmtr.substr(last_pos);
    if (partial) {
        std::replace(out.begin(), out.end(), '(', '{');
        std::replace(out.begin(), out.end(), ')', '}');
    }
    return out;
}

std::string FormatterHelper::applyFormat(const std::string &fmtr, const json::json &kwargs) {
    return FormatterHelper::applyFormat(fmtr, kwargs, false);
}


ArchiverOpts::ArchiverOpts(const std::string &key, const std::string &cam_fov, const std::string &mode) :
        key{key}, cam_fov{cam_fov}, mode{mode} {}

ArchiverOpts ArchiverOpts::from_env() {
    const char *env_fov = std::getenv("CAM_FOV");
    const char *env_mode = std::getenv("CAM_MODE");
    const char *env_key = std::getenv("ARCH_KEY");
    if (!(env_fov && env_fov[0])) {
        throw std::invalid_argument("ENV variable CAM_FOV must be specified");
    }
    if (!(env_mode && env_mode[0])) {
        throw std::invalid_argument("ENV variable CAM_MODE must be specified");
    }
    if (!(env_key && env_key[0])) {
        throw std::invalid_argument("ENV variable ARCH_KEY must be specified");
    }

    return ArchiverOpts{env_key, env_fov, env_mode};
}

ArchiverFormatter::ArchiverFormatter(const std::string &redis_uri,
                                     const std::string &key, const std::string &cam_fov, const std::string &mode) :
        rc{redis_uri},
        _key{key},
        cam_fov{cam_fov},
        mode{mode} {}

ArchiverFormatter::ArchiverFormatter(const std::string &redis_host, const std::string &redis_port,
                                     const std::string &key, const std::string &cam_fov, const std::string &mode) :
        rc{"tcp://" + redis_host + ":" + redis_port},
        _key{key},
        cam_fov{cam_fov},
        mode{mode} {}

ArchiverFormatter ArchiverFormatter::from_env() {
    const char *env_fov = std::getenv("CAM_FOV");
    const char *env_mode = std::getenv("CAM_MODE");
    const char *env_key = std::getenv("ARCH_KEY");
    if (!(env_fov && env_fov[0])) {
        throw std::invalid_argument("ENV variable CAM_FOV must be specified");
    }
    if (!(env_mode && env_mode[0])) {
        throw std::invalid_argument("ENV variable CAM_MODE must be specified");
    }
    if (!(env_key && env_key[0])) {
        throw std::invalid_argument("ENV variable ARCH_KEY must be specified");
    }
    const std::string fov{env_fov};
    const std::string mode{env_mode};
    const std::string key{env_key};
    const std::string redis_uri = RedisHelper::get_redis_uri();

    return ArchiverFormatter{redis_uri, key, fov, mode};
}

json::json ArchiverFormatter::fetchArchiveKwargs(redis::Redis &rc, const std::string &key, const std::string &cam_fov,
                                                 const std::string &mode) {
    json::json kwargs = RedisHelper::get_dict(rc, key);
    kwargs["cam_fov"] = cam_fov;
    kwargs["mode"] = mode;
    unsigned char cf = std::toupper(cam_fov[0]);
    kwargs["cf"] = std::string(1, cf);
    return kwargs;
}

bool ArchiverFormatter::fetchKwargs() {
    _kwargs = ArchiverFormatter::fetchArchiveKwargs(rc, _key, cam_fov, mode);
    return true;
}

std::string ArchiverFormatter::generateFilename(const std::chrono::system_clock::time_point &tp, const std::string &ext) {
    fetchKwargs();
    std::string fmtr = _kwargs["template"];
    auto duration = tp.time_since_epoch();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(duration) -
                 std::chrono::duration_cast<std::chrono::nanoseconds>(secs);
    std::string stime = isoformat_u(static_cast<time_t>(secs.count()), nsecs.count());
    _kwargs["time"] = stime;
    _kwargs["ext"] = ext;
    auto tmp = FormatterHelper::conformKwargsToFormatter(fmtr, _kwargs);
    return FormatterHelper::applyFormat(fmtr, tmp);
}

std::string ArchiverFormatter::generateFilename(const std::string &ext) {
    return generateFilename(std::chrono::system_clock::now(), ext);
}

json::json ArchiverFormatter::kwargs() {
    return _kwargs;
}

namespace ArchiverHelper {
    std::string generateFilename(std::shared_ptr<RedisEnvoy> envoy, const ArchiverOpts &opts,
                                 long int sec, long int nsec) {

        json::json kwargs = fetchArchiveKwargs(std::move(envoy), opts);
        std::string stime = isoformat_u(sec, nsec);
        std::string fmtr = kwargs["template"];
        RedisHelper::assert_is_not_null(opts.key + "template", fmtr);

//        std::cout << "fmtr " << fmtr << std::endl;
        std::string ext_key = "ext_" + opts.mode;
        json::json j_ext = kwargs[ext_key];
        RedisHelper::assert_is_not_null(ext_key, j_ext);
//        std::cout << "jext " << j_ext << std::endl;
        std::string ext = std::string(j_ext);
//        std::cout << "ext " << ext << std::endl;

        //stime[21] = '0';
        kwargs["time"] = stime;
        kwargs["ext"] = ext;
        auto tmp = FormatterHelper::conformKwargsToFormatter(fmtr, kwargs);
//        std::cout << "tmp kwargs " << tmp.dump() << std::endl;
        return FormatterHelper::applyFormat(fmtr, tmp);

    }
    json::json fetchArchiveKwargs(std::shared_ptr<RedisEnvoy> envoy, const ArchiverOpts &opts) {
//        std::cout << "before kwargs " << std::endl;
        json::json kwargs = envoy->get_dict(opts.key);
        kwargs["cam_fov"] = opts.cam_fov;
        kwargs["mode"] = opts.mode;
        unsigned char cf = std::toupper(opts.cam_fov[0]);
        kwargs["cf"] = std::string(1, cf);
//        std::cout << "kwargs " << kwargs.dump() << std::endl;
        return kwargs;
    }

    int get_is_archiving(std::shared_ptr<RedisEnvoy> envoy, const std::string &key) {
        std::string tmp = envoy->get(key);
//        std::cout << "get_is_arch " <<  tmp << std::endl;
        json::json tmpj = json::json::parse(tmp);
        if (!tmpj.is_primitive() || tmpj.is_null()) {
            throw std::invalid_argument("Redis Key " + key + " must exist and be of primitive type");
        }
        if (tmpj.is_number_integer()) {
            return (int) tmpj.front();
        } else if (tmpj.is_boolean()) {
            bool b = tmpj.front();
            return (int) b;
        } else if (tmpj.is_number_float()) {
            double x = tmpj.front();
            return x != 0;
        }
        throw std::invalid_argument("Redis Key " + key + " is invalid type");

    }


}
