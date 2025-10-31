
import vector1 from "../../assests/vector1.png";

const bgstyle2 = {
  backgroundImage: `url(${vector1})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const About = () => {
  return (
    <>
     <div  className="relative py-14 ">
      <div className="container py-4">
        <div>
          <div className="text-center text-4xl font-bold text-cyan-950">
            Why Choose Us
          </div>
          {/* column 1*/}
          <div className="pt-10">
            <div className="grid grid-cols-1 sm:grid-cols-2 place-items-center gap-4">
              <div
                style={bgstyle2}
                className="rounded-3xl overflow-hidden shadow-2xl"
              >
                <img src={vector1} alt="" className="w-full " />
              </div>
              <div>
                <div className="text-3xl font-semibold text-cyan-950 ">
                  Smart Scanning
                </div>
                <p className="py-4 font-semibold text-cyan-950">
                  Our innovative smart scanning feature uses your device's
                  camera to identify waste items quickly and accurately. Simply
                  point, scan, and let the app do the rest. It's designed for
                  effortless use, ensuring you get the right disposal guidance
                  in seconds.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </>
  )
}

export default About